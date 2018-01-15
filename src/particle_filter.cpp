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

using namespace std;

// Random generator engine
static default_random_engine rand_gen;

struct particle_ex : std::exception
{
	string error;
	const char *what() const noexcept { return error.c_str(); }

	particle_ex(string err) : error(err) {}
};

void ParticleFilter::init(double x, double y, double theta, double std[])
{
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	if (is_initialized == true)
	{
		return;
	}

	num_particles = 100;

	double std_x = std[0];
	double std_y = std[1];
	double std_theta = std[2];

	// Creating normal distributions
	normal_distribution<double> x_init_dist(x, std_x);
	normal_distribution<double> y_init_dist(y, std_y);
	normal_distribution<double> theta_init_dist(theta, std_theta);

	// Generate particles with normal distribution with mean on GPS values.
	for (int i = 0; i < num_particles; ++i)
	{
		Particle particle;

		particle.id = i;
		particle.x = x_init_dist(rand_gen);
		particle.y = y_init_dist(rand_gen);
		particle.theta = theta_init_dist(rand_gen);
		particle.weight = 1.0;

		particles.push_back(particle);
	}
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate)
{
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	if (is_initialized != true)
	{
		throw particle_ex("not initialized");
	}

	double std_x = std_pos[0];
	double std_y = std_pos[1];
	double std_theta = std_pos[2];

	// Creating noise normal distributions
	normal_distribution<double> noise_x(0, std_x);
	normal_distribution<double> noise_y(0, std_y);
	normal_distribution<double> noise_theta(0, std_theta);

	for (Particle &curr_particle : particles)
	{

		if (yaw_rate == 0)
		{
			// Add measurements to particles
			curr_particle.x += velocity * delta_t * cos(curr_particle.theta);
			curr_particle.y += velocity * delta_t * sin(curr_particle.theta);
			//curr_particle.theta = curr_particle.theta; // remains unchanged since the yaw rate is zero
		}
		else
		{
			// Add measurements to particles
			curr_particle.x += (velocity / yaw_rate) * (sin(curr_particle.theta + (yaw_rate * delta_t)) - sin(curr_particle.theta));
			curr_particle.y += (velocity / yaw_rate) * (cos(curr_particle.theta) - cos(curr_particle.theta + (yaw_rate * delta_t)));
			curr_particle.theta += yaw_rate * delta_t;
		}

		// Add noise to the particles
		curr_particle.x += noise_x(rand_gen);
		curr_particle.y += noise_y(rand_gen);
		curr_particle.theta += noise_theta(rand_gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs> &observations)
{
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	if (is_initialized != true)
	{
		throw particle_ex("not initialized");
	}

	for (LandmarkObs &obs : observations)
	{
		double min_sq_dist = numeric_limits<double>::max();

		for (LandmarkObs &pred : predicted)
		{
			double x = obs.x - pred.x;
			double y = obs.y - pred.y;

			double sq_dist = (x * x) + (y * y);

			if (sq_dist < min_sq_dist)
			{
				min_sq_dist = sq_dist;
				obs.id = pred.id;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
								   const std::vector<LandmarkObs> &observations, const Map &map_landmarks)
{
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

	if (is_initialized != true)
	{
		throw particle_ex("not initialized");
	}

	double sensor_range_sq = sensor_range * sensor_range;
	double gauss_norm = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);

	// The denominators of the mvGd also stay the same
	double x_denom = 2 * std_landmark[0] * std_landmark[0];
	double y_denom = 2 * std_landmark[1] * std_landmark[1];

	// Iterate through each particle
	for (int i = 0; i < num_particles; ++i)
	{

		// create and populate a copy of the list of observations transformed from vehicle coordinates to map coordinates
		vector<LandmarkObs> transformed_obs(observations.size());
		for (size_t j = 0; j < observations.size(); j++)
		{
			transformed_obs[j].id = observations[j].id;
			transformed_obs[j].x = cos(particles[i].theta) * observations[j].x - sin(particles[i].theta) * observations[j].y + particles[i].x;
			transformed_obs[j].y = sin(particles[i].theta) * observations[j].x + cos(particles[i].theta) * observations[j].y + particles[i].y;
		}

		vector<LandmarkObs> predictions;

		for (size_t j = 0; j < map_landmarks.landmark_list.size(); j++)
		{

			// get id and x,y coordinates
			float mx = map_landmarks.landmark_list[j].x_f;
			float my = map_landmarks.landmark_list[j].y_f;
			int m_id = map_landmarks.landmark_list[j].id_i;

			const double dx = mx - particles[i].x;
			const double dy = my - particles[i].y;
			const double dist_sq = dx * dx + dy * dy;

			// Use squares since sqrt is computationaly expensive
			if (dist_sq < sensor_range_sq)
			{

				predictions.push_back(LandmarkObs{m_id, mx, my});
			}
		}

		// perform dataAssociation for the predictions and transformed observations on current particle
		dataAssociation(predictions, transformed_obs);

		particles[i].weight = 1.0;

		for (unsigned int j = 0; j < transformed_obs.size(); j++)
		{

			// placeholders for observation and associated prediction coordinates
			double o_x, o_y, pr_x, pr_y;
			o_x = transformed_obs[j].x;
			o_y = transformed_obs[j].y;

			// get the x,y coordinates of the prediction associated with the current observation
			for (unsigned int k = 0; k < predictions.size(); k++)
			{
				if (predictions[k].id == transformed_obs[j].id)
				{
					pr_x = predictions[k].x;
					pr_y = predictions[k].y;
				}
			}

			// calculate weight for this observation with multivariate Gaussian
			double s_x = std_landmark[0];
			double s_y = std_landmark[1];

			// Calculate multi-variate Gaussian distribution
			double x_diff = o_x - pr_x;
			double y_diff = o_y - pr_y;
			double exponent = ((x_diff * x_diff) / x_denom) + ((y_diff * y_diff) / y_denom);

			particles[i].weight *= gauss_norm * exp(-exponent);
		}
	}
}

void ParticleFilter::resample()
{
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	if (is_initialized != true)
	{
		throw particle_ex("not initialized");
	}

	vector<Particle> new_particles(num_particles);
	vector<double> weights(num_particles);

	for (size_t i = 0; i < num_particles; ++i)
	{
		weights[i] = particles[i].weight;
	}

	for (int i = 0; i < num_particles; ++i)
	{
		discrete_distribution<int> weight_distru(weights.begin(), weights.end());
		new_particles[i] = particles[weight_distru(rand_gen)];
	}

	particles = new_particles;
}

Particle ParticleFilter::SetAssociations(Particle &particle, const std::vector<int> &associations,
										 const std::vector<double> &sense_x, const std::vector<double> &sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	particle.associations = associations;
	particle.sense_x = sense_x;
	particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<int>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
	copy(v.begin(), v.end(), ostream_iterator<float>(ss, " "));
	string s = ss.str();
	s = s.substr(0, s.length() - 1); // get rid of the trailing space
	return s;
}
