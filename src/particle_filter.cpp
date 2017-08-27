/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"
#include "map.h"

using namespace std;

#define PI 3.14159265

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
	ParticleFilter::num_particles = 333;

	default_random_engine gen;
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

	 std_x = std[0];
	 std_y = std[1];
	 std_theta = std[2];

    cout << "Initial value = " << x << "," << y << "," << theta << endl;
	// Create a normal (Gaussian) distribution for x, y and theta
	normal_distribution<double> dist_x(x, std_x);
	normal_distribution<double> dist_y(y, std_y);
	normal_distribution<double> dist_theta(theta, std_theta);

	for (int i = 0; i < num_particles; ++i) {
		Particle p;
        p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		ParticleFilter::particles.push_back(p);
		//cout << "Init particle co-ordinates = " << p.id << "," << p.x << "," << p.y << "," << p.theta << endl;
	}
	ParticleFilter::is_initialized = true;

	return;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine gen;
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

	 std_x = std_pos[0];
	 std_y = std_pos[1];
	 std_theta = std_pos[2];




	std::vector<Particle> temp_particles;
	for (int i = 0; i < num_particles; ++i) {
        Particle p = ParticleFilter::particles[i];

        // Create a normal (Gaussian) distribution for x, y and theta
        normal_distribution<double> dist_x(p.x, std_x);
        normal_distribution<double> dist_y(p.y, std_y);
        normal_distribution<double> dist_theta(p.theta, std_theta);

        /*
		Particle p = ParticleFilter::particles[i];
		p.x = p.x +
		      (velocity / yaw_rate) * (sin((p.theta + (yaw_rate * delta_t)) * PI/180) - sin(p.theta * PI/180)) +
		      dist_x(gen);
		p.y = p.y +
		      (velocity / yaw_rate) * (cos(p.theta * PI/180) - cos((p.theta + (yaw_rate * delta_t)) * PI/180)) +
		      dist_y(gen);
		p.theta = p.theta + (yaw_rate * delta_t) + dist_theta(gen);
        */
		p.x = (velocity / yaw_rate) * (sin((p.theta + (yaw_rate * delta_t))) - sin(p.theta)) + dist_x(gen);
		p.y = (velocity / yaw_rate) * (cos(p.theta) - cos((p.theta + (yaw_rate * delta_t)))) + dist_y(gen);
		p.theta = (yaw_rate * delta_t) + dist_theta(gen);

        /*
        p.x += (velocity / yaw_rate) * (sin((p.theta + (yaw_rate * delta_t))) - sin(p.theta));
		p.y += (velocity / yaw_rate) * (cos(p.theta) - cos((p.theta + (yaw_rate * delta_t))));
		p.theta += (yaw_rate * delta_t);
        */
		temp_particles.push_back(p);
	}
	ParticleFilter::particles = temp_particles;
    // cout << "Finishing Prediction" << endl;
    return;
}

std::vector<LandmarkObs> ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<Map::single_landmark_s> landmarks_list) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	//iterate through transformed observations to find closest landmark
	std::vector<LandmarkObs> closest_nei;
	for (int i = 0; i < predicted.size(); ++i) {
        LandmarkObs cn; // to track closest neighbor
        cn.id = 0;
	    double closest_distance = 99999.0;
        for (int j = 0; j < landmarks_list.size(); ++j) {
            double nei_distance = dist(predicted[i].x, predicted[i].y, landmarks_list[j].x_f, landmarks_list[j].y_f);
            if (nei_distance < closest_distance) {
                closest_distance = nei_distance;
                cn.id = landmarks_list[j].id_i;
                cn.x = landmarks_list[j].x_f;
                cn.y = landmarks_list[j].y_f;
            }
        }

        closest_nei.push_back(cn);
	}
	return closest_nei;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    std::vector<Particle> temp_particles;
	for (int i = 0; i < num_particles; ++i) {

		Particle p = ParticleFilter::particles[i];
		p.associations.clear();
		p.sense_x.clear();
		p.sense_y.clear();

        // filter out landmarks that are out of range
        std::vector<Map::single_landmark_s> inrange_landmarks;
        for (int i = 0; i < map_landmarks.landmark_list.size(); ++i) {
            Map::single_landmark_s lm = map_landmarks.landmark_list[i];
            double landmark_distance = dist(p.x, p.y, lm.x_f, lm.y_f);
            inrange_landmarks.push_back(lm);
            /*
            if (landmark_distance <= sensor_range) {
                inrange_landmarks.push_back(lm);
            }
            */
        }

        // transform landmark observations from the vehicle's coordinates to the map's coordinates, with respect to our particle.
        std::vector<LandmarkObs> transformed_observations;
        for(int i = 0; i < observations.size(); i++)
        {
            LandmarkObs obs = observations[i];
            LandmarkObs trans_obs;
            /*
            trans_obs.x = p.x + (cos(p.theta * PI/180) * obs.x) - (sin(p.theta * PI/180) * obs.y);
            trans_obs.y = p.y + (sin(p.theta * PI/180) * obs.x) + (cos(p.theta * PI/180) * obs.y);
            */
            trans_obs.x = p.x + (cos(p.theta) * obs.x) - (sin(p.theta) * obs.y);
            trans_obs.y = p.y + (sin(p.theta) * obs.x) + (cos(p.theta) * obs.y);
            transformed_observations.push_back(trans_obs);
        }

        std::vector<LandmarkObs> associated_landmarks = dataAssociation(transformed_observations, inrange_landmarks);

        double final_weight = 1.0;
        ParticleFilter::max_weight = 0.0;
        for (int i = 0; i < transformed_observations.size(); ++i) {

          double x_var = pow((transformed_observations[i].x - associated_landmarks[i].x),2);
          double y_var = pow((transformed_observations[i].y - associated_landmarks[i].y),2);

          // calculate normalization term
          double gauss_norm = (1/(2 * PI * std_landmark[0] * std_landmark[1]));

          // calculate exponent
          double exponent = (pow(x_var,2)/(2 * pow(std_landmark[0],2))) + (pow(y_var,2)/(2 * pow(std_landmark[1],2)));

          // calculate weight using normalization terms and exponent
          double weight = gauss_norm * exp(-exponent);

          //cout << "weight = " << weight << endl;
          final_weight = final_weight * weight;

          //set associations and corresponding x,y values
          p.associations.push_back(associated_landmarks[i].id);
          p.sense_x.push_back(transformed_observations[i].x);
          p.sense_y.push_back(transformed_observations[i].y);
        }
        p.weight = final_weight;
		temp_particles.push_back(p);
		if (final_weight > max_weight) {
            ParticleFilter::max_weight = final_weight;
		}
	}
	ParticleFilter::particles = temp_particles;
	//cout << "Best weight =" << ParticleFilter::max_weight << endl;
    // cout << "Finishing updateWeights" << endl;
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::random_device rd; // obtain a random number from hardware
    std::mt19937 eng(rd()); // seed the generator
    std::uniform_int_distribution<> distr(0, num_particles-1); // define the range
    std::uniform_int_distribution<> beta_distr(0, 100); // define the range
    int index = distr(eng);
    double beta = 0.0;
    std::vector<Particle> temp_particles;
	for (int i = 0; i < num_particles; ++i) {
        double beta_rand = beta_distr(eng) / 100.0;
        beta = beta + (beta_rand * 2 * ParticleFilter::max_weight);
        while(beta > ParticleFilter::particles[index].weight) {
            beta = beta - ParticleFilter::particles[index].weight;
            index = (index + 1) % num_particles;
        }
        Particle p = ParticleFilter::particles[index];
        temp_particles.push_back(p);
	}
	ParticleFilter::particles = temp_particles;
    // cout << "Finishing resample" << endl;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
