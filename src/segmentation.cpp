#include <ros/ros.h>

#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_types.h>
#include <pcl/PCLPointCloud2.h>
#include <pcl/conversions.h>
#include <pcl_ros/transforms.h>
#include <pcl_ros/point_cloud.h>

#include <pcl/ModelCoefficients.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/sample_consensus/method_types.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>

#include <visualization_msgs/Marker.h>

#include <jsk_recognition_msgs/BoundingBox.h>
#include <jsk_recognition_msgs/BoundingBoxArray.h>

#include <octomap_msgs/Octomap.h>
#include <octomap_msgs/GetOctomap.h>
#include <octomap_msgs/BoundingBoxQuery.h>
#include <octomap_msgs/conversions.h>

#include <octomap_ros/conversions.h>
#include <octomap/octomap.h>
#include <octomap/OcTreeKey.h>
#include <octomap/math/Pose6D.h>

#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv2/contrib/contrib.hpp>

#include <limits>
#include <cmath>

#include <tf/tf.h>

using namespace cv;

std::vector<cv::Scalar> _colors;
ros::Publisher pub_cluster_cloud;
ros::Publisher pub_jsk_boundingboxes;
ros::Publisher pub_worldmap_cloud;
ros::Publisher pub_nomap_points_cloud;

tf::StampedTransform* _transform;
tf::TransformListener* _transform_listener;

pcl::PointCloud<pcl::PointXYZ> _worldmap_cloud;
pcl::PointCloud<pcl::PointXYZ> _sensor_cloud;

octomap::OcTree *_map_octotree_ptr;

std_msgs::Header _ndt_header;

bool _map_transformed;
bool _using_sensor_cloud;

void publishCloud(ros::Publisher* in_publisher, pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_to_publish_ptr)
{
	sensor_msgs::PointCloud2 cloud_msg;
	pcl::toROSMsg(*in_cloud_to_publish_ptr, cloud_msg);
	in_publisher->publish(cloud_msg);
}

void publishColorCloud(ros::Publisher* in_publisher, pcl::PointCloud<pcl::PointXYZRGB>::Ptr in_cloud_to_publish_ptr)
{
	sensor_msgs::PointCloud2 cloud_msg;
	pcl::toROSMsg(*in_cloud_to_publish_ptr, cloud_msg);
	in_publisher->publish(cloud_msg);
}

void map_callback (const sensor_msgs::PointCloud2ConstPtr& in_map_cloud)
{
	pcl::PointCloud<pcl::PointXYZ> map_cloud;
	pcl::fromROSMsg(*in_map_cloud, map_cloud);

	try
	{
		_transform_listener->lookupTransform("/world", "/map",	 ros::Time(in_map_cloud->header.stamp), *_transform);
		if(map_cloud.points.size()<=0)
		{
			ROS_ERROR("The input cloud is empty, please make sure the node is subscribed to the correct PointCloud Map Topic.");
			return;
		}
		pcl_ros::transformPointCloud(map_cloud, 	//velodyne points in velodyne frame coords
										_worldmap_cloud, //velodyne points in map frame coords
										*_transform);
		_worldmap_cloud.header.frame_id="world";
		publishCloud(&pub_worldmap_cloud, _worldmap_cloud.makeShared());
		ROS_INFO("Map published as points_world in world frame");
		std::cout << _worldmap_cloud.points[0] << std::endl;
		std::cout << _worldmap_cloud.points.size() << std::endl;
	}
	catch (tf::TransformException &ex)
	{
		ROS_ERROR("TransformException in map_callback: %s",ex.what());
	}
}

void removeFloor(pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud_ptr, float in_max_height=0.2)
{
	pcl::SACSegmentation<pcl::PointXYZ> seg;
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);

	seg.setOptimizeCoefficients (true);
	seg.setModelType (pcl::SACMODEL_PLANE);
	seg.setMethodType (pcl::SAC_RANSAC);
	seg.setMaxIterations (100);
	seg.setDistanceThreshold (in_max_height);//floor distance
	seg.setOptimizeCoefficients(true);
	seg.setInputCloud(in_cloud_ptr);
	seg.segment(*inliers, *coefficients);
	if (inliers->indices.size () == 0)
	{
		std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
	}

	/*REMOVE THE FLOOR FROM THE CLOUD*/
	pcl::ExtractIndices<pcl::PointXYZ> extract;
	extract.setInputCloud (in_cloud_ptr);
	extract.setIndices(inliers);
	extract.setNegative(true);//true removes the indices, false leaves only the indices
	extract.filter(*out_cloud_ptr);
}

void octomap_callback(const octomap_msgs::Octomap in_octomap)
{
	_map_octotree_ptr = dynamic_cast<octomap::OcTree*> (octomap_msgs::msgToMap(in_octomap));
	if(_map_octotree_ptr!=NULL)
	{
		ROS_INFO("Octree read from OctomapServer with %d nodes", _map_octotree_ptr->getNumLeafNodes());
	}
	else
	{
		ROS_ERROR("Invalid Octree received from OctomapServer");
	}
}

void transformAndFilterMap(pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr, pcl::PointCloud<pcl::PointXYZ>::Ptr out_cloud_ptr, geometry_msgs::PoseStamped in_localizer_pose_msg, float in_nearest_neighbor_radius=0.3, float in_map_distance_threshold=50)
{
	pcl::PointCloud<pcl::PointXYZ> map_cloud_transformed;
	pcl::PointCloud<pcl::PointXYZ> velodyne_transformed;
	try
	{
		////////////////TEST CODE TO CREATE OCTOMAP USING Velodyne DATA INSTEAD OF LATCHED MAP///////////
		/*pcl::PointCloud<pcl::PointXYZ> absolute_cloud_transformed;
		_transform_listener->lookupTransform("/map", "/velodyne",
						ros::Time(in_localizer_pose_msg.header.stamp), *_transform);

		pcl_ros::transformPointCloud(*in_cloud_ptr, absolute_cloud_transformed, *_transform);

		absolute_cloud_transformed.header.frame_id="map";

		publishCloud(&pub_absolute_points_cloud, absolute_cloud_transformed.makeShared());

		return;*/
		////////////////////////////////////

		///FROM HERE INSTEAD OF SUBSCRIBING TO points_map WE WILL GET THE ROUGH DEFINED MAP BY OCTOMAP

		//Workflow
		//1. Check if Octomap has been read
		////
		//ROS_INFO("Octree available with %d nodes", _map_octotree_ptr->getNumLeafNodes());
		if (_map_octotree_ptr== NULL || _map_octotree_ptr->getNumLeafNodes()<=0)
		{
			ROS_ERROR("transformAndFilterMap: The OctoMap is not avaialble, please make sure the OctoMapServer is running and it's subscribing to the correct PointCloud Map Topic.");
			out_cloud_ptr=in_cloud_ptr;
			return;
		}
		//2. Transform Velodyne points' coords to map frame
		_transform_listener->lookupTransform("/world", "/velodyne",
				ros::Time(_ndt_header.stamp), *_transform);

		if(in_cloud_ptr->points.size()<=0)
		{
			ROS_ERROR("transformAndFilterMap: The input cloud is empty, please make sure the node is subscribed to the correct PointCloud Map Topic.");
			out_cloud_ptr=in_cloud_ptr;
			return;
		}
		pcl_ros::transformPointCloud(*in_cloud_ptr, 	//velodyne points in velodyne frame coords
										velodyne_transformed, //velodyne points in map frame coords
										*_transform);

		velodyne_transformed.header.frame_id="world";//change also frame to map

		//3. get each velodyne point in map's coords and check for each octomap voxel if its "occupied"
		std::vector<bool> found_point_index(velodyne_transformed.points.size(), false);//vector to store the indices of the points found in an occupied octomap voxel
		for (unsigned int i = 0; i < velodyne_transformed.points.size(); i++)
		{
			pcl::PointXYZ current_point;
			current_point.x=velodyne_transformed.points[i].x;
			current_point.y=velodyne_transformed.points[i].y;
			current_point.z=velodyne_transformed.points[i].z;

			octomap::OcTreeNode* node_ptr= _map_octotree_ptr->search(current_point.x, current_point.y, current_point.z);
			if (node_ptr!=NULL && _map_octotree_ptr->isNodeOccupied(node_ptr))
			{
				found_point_index[i] = true;
			}
		}
		//	4. if it is, add the velodyne point's index to the std::vector holding the indices to be deleted
		pcl::PointIndices::Ptr found_indices (new pcl::PointIndices);
		for(unsigned int i=0; i<velodyne_transformed.points.size();i++)//get the indices of the duplicates
		{
			if(found_point_index[i])
			{
				found_indices->indices.push_back(i);
			}
		}
		//5. delete them using the filter
		//if (duplicates_indices->indices.size()>0)//points matched proceed to remove them
		{
			//std::cout << "Duplicates found " << duplicates_indices->indices.size() ;
			pcl::ExtractIndices<pcl::PointXYZ> extract;
			extract.setInputCloud (velodyne_transformed.makeShared());
			extract.setIndices(found_indices);
			extract.setNegative(true);//true removes the indices, false leaves only the indices
			extract.filter(*out_cloud_ptr);
		}
		std::cout << " Original Points" << in_cloud_ptr->points.size() << " Points remaining " << out_cloud_ptr->points.size() << std::endl;

		/*if (_map_cloud.points.size()>0)
		{
			pcl_ros::transformPointCloud(_map_cloud, map_cloud_transformed, *_transform);
			if (!_map_transformed)
				ROS_INFO("Map correctly transformed");
			_map_transformed=true;

			map_cloud_transformed.header.frame_id="velodyne";

			publishCloud(&pub_map_cloud, map_cloud_transformed.makeShared());

			//Remove Map's Points from Velodyne ones
			//create KdTree
			pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
			kdtree.setInputCloud (in_cloud_ptr);

			std::vector<bool> found_point_index(in_cloud_ptr->points.size(), false);

			pcl::PointIndices::Ptr duplicates_indices (new pcl::PointIndices);
			std::vector<int> points_indices_search;
			std::vector<float> points_distances_search;
			for (unsigned int i=0; i< map_cloud_transformed.points.size(); i++)//seach on each point of the map and velodyne
			{
				pcl::PointXYZ current_point;
				current_point.x=map_cloud_transformed.points[i].x;
				current_point.y=map_cloud_transformed.points[i].y;
				current_point.z=map_cloud_transformed.points[i].z;

				if (abs(current_point.x) < in_map_distance_threshold &&
						abs(current_point.y) < in_map_distance_threshold &&
						abs(current_point.z) < in_map_distance_threshold)//only process points found in the map inside the defined threshold
				{
					if(kdtree.radiusSearch(current_point, in_nearest_neighbor_radius, points_indices_search, points_distances_search)>0)
					{
						for (unsigned int j=0; j<points_indices_search.size(); j++)//mark as duplicate each of the points in the radius
						{
							found_point_index[points_indices_search[j]]=true;
						}
					}
				}
			}
			for(unsigned int i=0; i<in_cloud_ptr->points.size();i++)//get the indices of the duplicates
			{
				if(found_point_index[i])
				{
					duplicates_indices->indices.push_back(i);
				}
			}
			if (duplicates_indices->indices.size()>0)//points matched proceed to remove them
			{
				//std::cout << "Duplicates found " << duplicates_indices->indices.size() ;
				pcl::ExtractIndices<pcl::PointXYZ> extract;
				extract.setInputCloud (in_cloud_ptr);
				extract.setIndices(duplicates_indices);
				extract.setNegative(true);//true removes the indices, false leaves only the indices
				extract.filter(*out_cloud_ptr);

				//std::cout << " Original Points" << in_cloud_ptr->points.size() << " Points remaining " << out_cloud_ptr->points.size() << std::endl;
			}
			else
			{
				ROS_INFO("No duplicates found between PointClouds");
			}
		}
		else
		{
			ROS_ERROR("Map Cloud is empty");
			_map_transformed=false;
			//cloud1 = pre_cloud1;
		}*/
	}
	catch (tf::TransformException &ex)
	{
		ROS_ERROR("%s",ex.what());
		_map_transformed=false;
		out_cloud_ptr = in_cloud_ptr;
	}
}

void clusterAndColor(pcl::PointCloud<pcl::PointXYZ>::Ptr in_cloud_ptr, pcl::PointCloud<pcl::PointXYZRGB>::Ptr out_cloud_ptr, double in_max_cluster_distance=0.5)
{
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);

	tree->setInputCloud (in_cloud_ptr);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance (in_max_cluster_distance); //
	ec.setMinClusterSize (10);
	ec.setMaxClusterSize (500);
	ec.setSearchMethod(tree);
	ec.setInputCloud (in_cloud_ptr);
	ec.extract (cluster_indices);

	/////////////////////////////////
	//---	3. Color clustered points
	/////////////////////////////////
	int j = 0;
	unsigned int k = 0;
	//pcl::PointCloud<pcl::PointXYZRGB>::Ptr final_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);

	jsk_recognition_msgs::BoundingBoxArray boundingbox_array;
	boundingbox_array.header = _ndt_header;
	boundingbox_array.header.frame_id="world";

	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);//coord + color cluster

		//assign color to each cluster
		for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
		{
			//fill new colored cluster point by point
			pcl::PointXYZRGB p;
			p.x = in_cloud_ptr->points[*pit].x;
			p.y = in_cloud_ptr->points[*pit].y;
			p.z = in_cloud_ptr->points[*pit].z;
			p.r = _colors[k].val[0];
			p.g = _colors[k].val[1];
			p.b = _colors[k].val[2];

			float origin_distance=sqrt(p.x*p.x + p.y*p.y);
			if ( origin_distance > 0.2 )
			{
				cloud_cluster->points.push_back(p);
			}
		}

		//get min, max
		float min_x=std::numeric_limits<float>::max();
		float max_x=-std::numeric_limits<float>::max();
		float min_y=std::numeric_limits<float>::max();
		float max_y=-std::numeric_limits<float>::max();
		float min_z=std::numeric_limits<float>::max();
		float max_z=-std::numeric_limits<float>::max();

		for(unsigned int i=0; i<cloud_cluster->points.size();i++)
		{
			if(cloud_cluster->points[i].x<min_x)
				min_x = cloud_cluster->points[i].x;
			if(cloud_cluster->points[i].y<min_y)
				min_y = cloud_cluster->points[i].y;
			if(cloud_cluster->points[i].z<min_z)
				min_z = cloud_cluster->points[i].z;

			if(cloud_cluster->points[i].x>max_x)
				max_x = cloud_cluster->points[i].x;
			if(cloud_cluster->points[i].y>max_y)
				max_y = cloud_cluster->points[i].y;
			if(cloud_cluster->points[i].z>max_z)
				max_z = cloud_cluster->points[i].z;
		}

		pcl::PointXYZ min_point(min_x, min_y, min_z), max_point(max_x, max_y, max_z);

		float l = max_point.x - min_point.x;
		float w = max_point.y - min_point.y;
		float h = max_point.z - min_point.z;

		jsk_recognition_msgs::BoundingBox bounding_box;
		bounding_box.header = _ndt_header;
		bounding_box.header.frame_id="world";

		bounding_box.pose.position.x = min_point.x + l/2;
		bounding_box.pose.position.y = min_point.y + w/2;
		bounding_box.pose.position.z = min_point.z + h/2;

		bounding_box.dimensions.x = ((l<0)?-1*l:l);
		bounding_box.dimensions.y = ((w<0)?-1*w:w);
		bounding_box.dimensions.z = ((h<0)?-1*h:h);

		//pose estimation for the cluster
		//test using linear regression
		//Slope(b) = (NΣXY - (ΣX)(ΣY)) / (NΣX2 - (ΣX)2)
		float sum_x=0, sum_y=0, sum_xy=0, sum_xx=0;
		for (unsigned int i=0; i<cloud_cluster->points.size(); i++)
		{
			sum_x+= cloud_cluster->points[i].x;
			sum_y+= cloud_cluster->points[i].y;
			sum_xy+= cloud_cluster->points[i].x*cloud_cluster->points[i].y;
			sum_xx+= cloud_cluster->points[i].x*cloud_cluster->points[i].x;
		}
		double slope= (cloud_cluster->points.size()*sum_xy - (sum_x*sum_y))/(cloud_cluster->points.size()*sum_xx - sum_x*sum_x);

		double rz = atan(slope);

		tf::Quaternion quat = tf::createQuaternionFromRPY(0.0, 0.0, 0.0);

		tf::quaternionTFToMsg(quat, bounding_box.pose.orientation);

		if (bounding_box.dimensions.x >0 && bounding_box.dimensions.y >0 && bounding_box.dimensions.z>0 &&
				bounding_box.dimensions.x < 5 && bounding_box.dimensions.y < 5 && bounding_box.dimensions.z < 5)
		{
			boundingbox_array.boxes.push_back(bounding_box);
		}

		cloud_cluster->width = cloud_cluster->points.size();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;

		*out_cloud_ptr = *out_cloud_ptr + *cloud_cluster;//sum up all the colored cluster into a complete pc

		j++; k++;
	}
	std::cout << "Clusters: " << k << std::endl;

	//---	4. Publish
	//convert back to ros
	pcl_conversions::toPCL(_ndt_header, out_cloud_ptr->header);
	out_cloud_ptr->header.frame_id="world";
	// Publish the data
	//pub_cluster_cloud.publish(final_cluster);
	pub_jsk_boundingboxes.publish(boundingbox_array);
}

void ndt_callback(const geometry_msgs::PoseStamped localizer_pose_msg)
{
	//perform calculations here
	if (!_using_sensor_cloud)
	{

		_using_sensor_cloud = true;
		_ndt_header = localizer_pose_msg.header;

		pcl::PointCloud<pcl::PointXYZ>::Ptr nofloor_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr nomap_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_clustered_cloud_ptr (new pcl::PointCloud<pcl::PointXYZRGB>);
		pcl::PointCloud<pcl::PointXYZ>::Ptr sensor_cloud_ptr = _sensor_cloud.makeShared();

		//Remove largest plane
		removeFloor(sensor_cloud_ptr, nofloor_cloud_ptr);
		//ROS_INFO ("After floor removal the point cloud contains: %lu", nofloor_cloud_ptr->points.size());

		_transform->setOrigin(tf::Vector3(localizer_pose_msg.pose.position.x,
								localizer_pose_msg.pose.position.y,
								localizer_pose_msg.pose.position.z));
		tf::Quaternion ndt_quaternion(localizer_pose_msg.pose.orientation.x,
								localizer_pose_msg.pose.orientation.y,
								localizer_pose_msg.pose.orientation.z,
								localizer_pose_msg.pose.orientation.w);

		_transform->setRotation(ndt_quaternion);

		//Remove points found in both the map and the sensor
		transformAndFilterMap(nofloor_cloud_ptr, nomap_cloud_ptr, localizer_pose_msg);

		publishCloud(&pub_nomap_points_cloud, nomap_cloud_ptr);

		clusterAndColor(nomap_cloud_ptr, colored_clustered_cloud_ptr);

		publishColorCloud(&pub_cluster_cloud, colored_clustered_cloud_ptr);


		//publishCloud(&pub_ndt_points_cloud, sensor_cloud_ptr);

		_using_sensor_cloud = false;
	}
}

void velodyne_callback(const sensor_msgs::PointCloud2ConstPtr& in_sensor_cloud)
{
	//store only the point cloud from velodyne and wait for ndt to finish
	float distance_threshold = 50;
	if (!_using_sensor_cloud)
	{
		pcl::PointCloud<pcl::PointXYZ>::Ptr current_sensor_cloud_ptr (new pcl::PointCloud<pcl::PointXYZ>);
		pcl::fromROSMsg(*in_sensor_cloud, *current_sensor_cloud_ptr);
		pcl::PointIndices::Ptr far_indices (new pcl::PointIndices);
		for(unsigned int i=0; i< current_sensor_cloud_ptr->points.size(); i++)
		{
			pcl::PointXYZ current_point;
			current_point.x=current_sensor_cloud_ptr->points[i].x;
			current_point.y=current_sensor_cloud_ptr->points[i].y;
			current_point.z=current_sensor_cloud_ptr->points[i].z;

			if (abs(current_point.x) >= distance_threshold ||
					abs(current_point.y) >= distance_threshold ||
					abs(current_point.z) >= 2.0)//only process points found in the map inside the defined threshold
			{
				far_indices->indices.push_back(i);
			}
		}

		_sensor_cloud.points.clear();

		pcl::ExtractIndices<pcl::PointXYZ> extract;
		extract.setInputCloud (current_sensor_cloud_ptr);
		extract.setIndices(far_indices);
		extract.setNegative(true);//true removes the indices, false leaves only the indices
		extract.filter(_sensor_cloud);
	}
}


/*
void cloud_callback (const sensor_msgs::PointCloud2ConstPtr& in_sensor_cloud)
{
	// Container for original & filtered data
	pcl::PCLPointCloud2* input_cloud = new pcl::PCLPointCloud2;
	pcl::PCLPointCloud2ConstPtr cloudPtr(input_cloud);

	// Convert to PCL data type
	pcl_conversions::toPCL(*in_sensor_cloud, *input_cloud);

	//Store PCL and PCL2 formats
	pcl::PointCloud<pcl::PointXYZ>::Ptr velo_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::PointCloud<pcl::PointXYZ>::Ptr preprocess_velo_cloud(new pcl::PointCloud<pcl::PointXYZ>);
	pcl::fromPCLPointCloud2(*input_cloud, *preprocess_velo_cloud);

	pcl::SACSegmentation<pcl::PointXYZ> seg;
	pcl::PointIndices::Ptr inliers (new pcl::PointIndices);
	pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients);
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_filtered (new pcl::PointCloud<pcl::PointXYZ>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr cloud_no_floor (new pcl::PointCloud<pcl::PointXYZ>());



	///////////////////////////////////////

	/////////////////////////////////
	//---	1. Remove planes (floor)
	/////////////////////////////////

	float distance = 0.5;//this may be a parameter
	seg.setOptimizeCoefficients (true);
	seg.setModelType (pcl::SACMODEL_PLANE);
	seg.setMethodType (pcl::SAC_RANSAC);
	seg.setMaxIterations (100);
	seg.setDistanceThreshold (0.2);//floor distance
	seg.setOptimizeCoefficients(true);
	seg.setInputCloud(preprocess_velo_cloud);
	seg.segment(*inliers, *coefficients);
	if (inliers->indices.size () == 0)
	{
		std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
	}
	else
	{
		//pcl::copyPointCloud(*cloud1, *inliers, *cloud_plane);
	}

	REMOVE THE FLOOR FROM THE CLOUD
		pcl::ExtractIndices<pcl::PointXYZ> extract;
		extract.setInputCloud (preprocess_velo_cloud);
		extract.setIndices(inliers);
		extract.setNegative(true);//true removes the indices, false leaves only the indices
		extract.filter(*velo_cloud);

	std::cout << "Points remaining after floor removal" << velo_cloud->points.size() << std::endl;

	int nr_points = (int) cloud_f->points.size ();
	while (cloud_f->points.size () > distance * nr_points)
	{
		// Segment the largest planar component from the remaining cloud
		seg.setInputCloud (cloud1);
		seg.segment (*inliers, *coefficients);
		if (inliers->indices.size () == 0)
		{
			std::cout << "Could not estimate a planar model for the given dataset." << std::endl;
			break;
		}

		// Extract the planar inliers from the input cloud
		pcl::ExtractIndices<pcl::PointXYZI> extract;
		extract.setInputCloud (cloud1);
		extract.setIndices(inliers);

		// Get the points associated with the planar surface
		//extract.setNegative(true);
		//extract.filter (*cloud_plane);

		// Remove the planar inliers, extract the rest
		extract.setNegative (true);
		extract.filter (*cloud_f);
		// *cloud1 = *cloud_f;
	}


			PUBLISH ONLY FLOOR
		pcl_conversions::toPCL(in_sensor_cloud->header, cloud_plane->header);
			// Publish the data
			pub.publish (cloud_f);
			return;


	CHANGE MAP COORDS TO MAP

	pcl::PointCloud<pcl::PointXYZ> map_cloud_transformed;
	float map_distance_threshold = 20;//50 meters
	float radius_threshold = 0.3;//5cm to match points between map and velodyne
	try
	{
		_transform_listener->lookupTransform("/velodyne", "/map",
				ros::Time(0), *_transform);
		if (_map_cloud.points.size()>0)
		{
			pcl_ros::transformPointCloud(_map_cloud, map_cloud_transformed, *_transform);
			if (!_map_transformed)
				ROS_INFO("Map correctly transformed");
			_map_transformed=true;

			map_cloud_transformed.header.frame_id="velodyne";
			//sensor_msgs::PointCloud2 map_cloud_transformed_msg;
			//pcl::toROSMsg(map_cloud_transformed, map_cloud_transformed_msg);
			//pub_map.publish(map_cloud_transformed);

			//Remove Map's Points from Velodyne ones
			//create KdTree
			pcl::KdTreeFLANN<pcl::PointXYZ> kdtree;
			kdtree.setInputCloud (velo_cloud);

			std::vector<bool> found_point_index(velo_cloud->points.size(), false);

			pcl::PointIndices::Ptr duplicates_indices (new pcl::PointIndices);
			std::vector<int> points_indices_search;
			std::vector<float> points_distances_search;
			for (unsigned int i=0; i< map_cloud_transformed.points.size(); i++)//seach on each point of the map and velodyne
			{
				pcl::PointXYZ current_point;
				current_point.x=map_cloud_transformed.points[i].x;
				current_point.y=map_cloud_transformed.points[i].y;
				current_point.z=map_cloud_transformed.points[i].z;

				if (abs(current_point.x) < map_distance_threshold &&
						abs(current_point.y) < map_distance_threshold &&
						abs(current_point.z) < map_distance_threshold)
				{
					if(kdtree.radiusSearch(current_point, radius_threshold, points_indices_search, points_distances_search)>0)
					{
						//duplicates_indices->indices.insert(duplicates_indices->indices.end(), points_indices_search.begin(), points_indices_search.end());
						//if(i==0)
						{
							for (unsigned int j=0; j<points_indices_search.size(); j++)
							{
								found_point_index[points_indices_search[j]]=true;
							}
						}
					}
				}
			}
			for(unsigned int i=0; i<velo_cloud->points.size();i++)
			{
				if(found_point_index[i])
				{
					duplicates_indices->indices.push_back(i);
				}
			}
			if (duplicates_indices->indices.size()>0)//points matched proceed to remove them
			{
				std::cout << "Duplicates found " << duplicates_indices->indices.size() ;
				pcl::ExtractIndices<pcl::PointXYZ> extract;
				extract.setInputCloud (velo_cloud);
				extract.setIndices(duplicates_indices);
				extract.setNegative(true);//true removes the indices, false leaves only the indices
				extract.filter(*cloud_no_floor);

				std::cout << " Original Points" << velo_cloud->points.size() << " Points remaining " << cloud_no_floor->points.size() << std::endl;
			}
			else
			{
				ROS_INFO("No duplicates found between PointClouds");
			}

		}
		else
		{
			ROS_ERROR("Map Cloud is empty");
			_map_transformed=false;
			//cloud1 = pre_cloud1;
		}
	}
	catch (tf::TransformException &ex)
	{
		ROS_ERROR("%s",ex.what());
		_map_transformed=false;
		preprocess_velo_cloud = velo_cloud;
	}

		//cloud_no_floor = velo_cloud;

	sensor_msgs::PointCloud2 cloud_filtered_msg;
	pcl::toROSMsg(*cloud_no_floor, cloud_filtered_msg);
	pub_map_cloud.publish(cloud_filtered_msg);



	pcl::VoxelGrid<pcl::PointXYZ> voxel_grid;
	voxel_grid.setInputCloud (cloud_no_floor);
	voxel_grid.setLeafSize (0.2, 0.2, 0.2);
	voxel_grid.filter(*cloud_filtered);
	//cloud_filtered = velo_cloud;

	//cloud_f = cloud1; //No floor removal
	/////////////////////////////////
	//---	2. Euclidean Clustering
	/////////////////////////////////

	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree (new pcl::search::KdTree<pcl::PointXYZ>);

	tree->setInputCloud (cloud_filtered);

	std::vector<pcl::PointIndices> cluster_indices;
	pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
	ec.setClusterTolerance (distance); //
	ec.setMinClusterSize (10);
	ec.setMaxClusterSize (500);
	ec.setSearchMethod(tree);
	ec.setInputCloud (cloud_filtered);
	ec.extract (cluster_indices);

	/////////////////////////////////
	//---	3. Color clustered points
	/////////////////////////////////
	int j = 0;
	unsigned int k = 0;
	pcl::PointCloud<pcl::PointXYZRGB>::Ptr final_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);

	jsk_recognition_msgs::BoundingBoxArray boundingbox_array;
	boundingbox_array.header = in_sensor_cloud->header;

	for (std::vector<pcl::PointIndices>::const_iterator it = cluster_indices.begin(); it != cluster_indices.end(); ++it)
	{
		pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud_cluster (new pcl::PointCloud<pcl::PointXYZRGB>);//coord + color cluster

		//assign color to each cluster
		for (std::vector<int>::const_iterator pit = it->indices.begin(); pit != it->indices.end(); ++pit)
		{
			//fill new colored cluster point by point
			pcl::PointXYZRGB p;
			p.x = cloud_filtered->points[*pit].x;
			p.y = cloud_filtered->points[*pit].y;
			p.z = cloud_filtered->points[*pit].z;
			p.r = _colors[k].val[0];
			p.g = _colors[k].val[1];
			p.b = _colors[k].val[2];

			float origin_distance=sqrt(p.x*p.x + p.y*p.y);
			//std::cout << "intensity" << cloud_plane->points[*pit].intensity << std::endl;
			if ( origin_distance > 0.2 )
			{
				cloud_cluster->points.push_back(p);

			}
		}

		//get min, max
		float min_x=std::numeric_limits<float>::max();
		float max_x=-std::numeric_limits<float>::max();
		float min_y=std::numeric_limits<float>::max();
		float max_y=-std::numeric_limits<float>::max();
		float min_z=std::numeric_limits<float>::max();
		float max_z=-std::numeric_limits<float>::max();

		for(unsigned int i=0; i<cloud_cluster->points.size();i++)
		{
			if(cloud_cluster->points[i].x<min_x)
				min_x = cloud_cluster->points[i].x;
			if(cloud_cluster->points[i].y<min_y)
				min_y = cloud_cluster->points[i].y;
			if(cloud_cluster->points[i].z<min_z)
				min_z = cloud_cluster->points[i].z;

			if(cloud_cluster->points[i].x>max_x)
				max_x = cloud_cluster->points[i].x;
			if(cloud_cluster->points[i].y>max_y)
				max_y = cloud_cluster->points[i].y;
			if(cloud_cluster->points[i].z>max_z)
				max_z = cloud_cluster->points[i].z;
		}

		pcl::PointXYZ min_point(min_x, min_y, min_z), max_point(max_x, max_y, max_z);

		float l = max_point.x - min_point.x;
		float w = max_point.y - min_point.y;
		float h = max_point.z - min_point.z;

		jsk_recognition_msgs::BoundingBox bounding_box;
		bounding_box.header = in_sensor_cloud->header;

		bounding_box.pose.position.x = min_point.x + l/2;
		bounding_box.pose.position.y = min_point.y + w/2;
		bounding_box.pose.position.z = min_point.z + h/2;

		bounding_box.dimensions.x = ((l<0)?-1*l:l);
		bounding_box.dimensions.y = ((w<0)?-1*w:w);
		bounding_box.dimensions.z = ((h<0)?-1*h:h);

		//pose estimation for the cluster
		//test using linear regression
		//Slope(b) = (NΣXY - (ΣX)(ΣY)) / (NΣX2 - (ΣX)2)
		float sum_x=0, sum_y=0, sum_xy=0, sum_xx=0;
		for (unsigned int i=0; i<cloud_cluster->points.size(); i++)
		{
			sum_x+= cloud_cluster->points[i].x;
			sum_y+= cloud_cluster->points[i].y;
			sum_xy+= cloud_cluster->points[i].x*cloud_cluster->points[i].y;
			sum_xx+= cloud_cluster->points[i].x*cloud_cluster->points[i].x;
		}
		double slope= (cloud_cluster->points.size()*sum_xy - (sum_x*sum_y))/(cloud_cluster->points.size()*sum_xx - sum_x*sum_x);

		double rz = atan(slope);

		tf::Quaternion quat = tf::createQuaternionFromRPY(0.0, 0.0, rz);

		tf::quaternionTFToMsg(quat, bounding_box.pose.orientation);

		//std::cout << min_point << "....." << max_point << std::endl;

		if (bounding_box.dimensions.x >0 && bounding_box.dimensions.y >0 && bounding_box.dimensions.z>0 &&
				bounding_box.dimensions.x < 5 && bounding_box.dimensions.y < 5 && bounding_box.dimensions.z < 5)
		{
			boundingbox_array.boxes.push_back(bounding_box);
		}

		cloud_cluster->width = cloud_cluster->points.size();
		cloud_cluster->height = 1;
		cloud_cluster->is_dense = true;

		*final_cluster = *final_cluster + *cloud_cluster;//sum up all the colored cluster into a complete pc

		j++; k++;
	}

	//---	4. Publish
	//convert back to ros
	pcl_conversions::toPCL(in_sensor_cloud->header, final_cluster->header);
	// Publish the data
	pub_cluster_cloud.publish(final_cluster);

	pub_jsk_boundingboxes.publish(boundingbox_array);

}
*/

int main (int argc, char** argv)
{
	// Initialize ROS
	ros::init (argc, argv, "euclidean_clustering");

	ros::NodeHandle h;
	ros::NodeHandle private_nh("~");

	cv::generateColors(_colors, 100);

	pub_cluster_cloud = h.advertise<sensor_msgs::PointCloud2>("/points_cluster",1);
	pub_jsk_boundingboxes = h.advertise<jsk_recognition_msgs::BoundingBoxArray>("/bounding_boxes",1);
	pub_nomap_points_cloud = h.advertise<sensor_msgs::PointCloud2>("/points_nomap",1);
	pub_worldmap_cloud = h.advertise<sensor_msgs::PointCloud2>("/points_world",1, true);


	tf::StampedTransform transform;
	tf::TransformListener listener;

	_transform = &transform;
	_transform_listener = &listener;
	_map_transformed = false;
	_using_sensor_cloud = false;

	std::string points_topic;

	if (private_nh.getParam("points_node", points_topic))
	{
		ROS_INFO("euclidean_clustering > Setting points node to %s", points_topic.c_str());
	}
	else
	{
		ROS_INFO("euclidean_clustering > No points node received, defaulting to velodyne_points, you can use _points_node:=YOUR_TOPIC");
		points_topic = "/vscan_points";
		points_topic = "/points_raw";
	}

	// Create a ROS subscriber for the input point cloud
	ros::Subscriber sub = h.subscribe (points_topic, 1, velodyne_callback);

	ros::Subscriber sub_map = h.subscribe ("/points_map", 1, map_callback);

	ros::Subscriber sub_ndt = h.subscribe ("localizer_pose", 1, ndt_callback);

	ros::Subscriber sub_octomap = h.subscribe ("octomap_full", 1, octomap_callback);


	// Spin
	ros::spin ();
}
