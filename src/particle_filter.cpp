/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>
#include <random>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;

static std::default_random_engine gen;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    // Set the number of particles
    num_particles = 100;

    // initialize particles
    for (int i = 0; i < num_particles; ++i) {
        Particle p;
        p.id = i;
        p.x = x;
        p.y = y;
        p.theta = theta;
        addNoise(p, std);
        p.weight = 1.0;
        particles.push_back(std::move(p));
    }

    // initialization finalized
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
    // Add measurements to each particle and add random Gaussian noise
    for (auto& p : particles) {
        if (fabs(yaw_rate) < 0.00001) {
            p.x += velocity * delta_t * cos(p.theta);
            p.y += velocity * delta_t * sin(p.theta);
        } else {
            p.x += velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
            p.y += velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
            p.theta += yaw_rate * delta_t;
        }
        addNoise(p, std_pos);
    }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> const &predicted,
                                     vector<LandmarkObs> &observations) {
    for (auto &o : observations) {
        // init minimum distance to maximum possible
        double min_dist = std::numeric_limits<double>::max();
        // init id of landmark
        int id = -1;
        for (auto const &p : predicted) {
            // distance between current and predicted landmark
            double curr_dist = dist(o.x, o.y, p.x, p.y);
            if (curr_dist < min_dist) {
                min_dist = curr_dist;
                id = p.id;
            }
        }
        // Assign the id of the nearest neighbor landmark to the transformed observation
        o.id = id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
    for (auto &p : particles) {

        // get predicted candidates
        vector<LandmarkObs> predictions;
        for (auto const &lm : map_landmarks.landmark_list) {
            // consider only those which are in the sensor range
            if (abs(dist(lm.x_f, lm.y_f, p.x, p.y)) <= sensor_range) {
                predictions.push_back(LandmarkObs{lm.id_i, lm.x_f, lm.y_f});
            }
        }

        // transform observation to map coordinate frame
        vector<LandmarkObs> observations_map;
        for (auto const &o : observations) {
            double x = cos(p.theta) * o.x - sin(p.theta) * o.y + p.x;
            double y = sin(p.theta) * o.x + cos(p.theta) * o.y + p.y;
            observations_map.push_back(LandmarkObs{o.id, x, y});
        }

        // perform data association on predictions and observations
        dataAssociation(predictions, observations_map);

        // re-initialize weights
        p.weight = 1.0;

        // update weights
        for (auto const &o_map : observations_map) {
            // get coordinates of landmark associated with prediction
            auto lm = std::find_if(predictions.begin(), predictions.end(),
                                   [&o_map](LandmarkObs const &lm) { return lm.id == o_map.id; });

            if (lm != predictions.end()) {
                double obs_weight = multivariate_gaussian_probability(o_map.x, o_map.y, lm->x, lm->y, std_landmark[0],
                                                                      std_landmark[1]);
                p.weight *= obs_weight;
            } else {
                throw std::runtime_error("could not find associated landmark object");
            }
        }
    }
}

void ParticleFilter::resample() {

    vector<double> weights;
    for (auto const& p : particles) {
        weights.push_back(p.weight);
    }

    std::discrete_distribution<> dist(weights.begin(), weights.end());

    vector<Particle> resampled;
    for (size_t i = 0; i < particles.size(); ++i) {
        resampled.push_back(particles[dist(gen)]);
    }

    particles = std::move(resampled);
}

void ParticleFilter::SetAssociations(Particle &particle,
                                     const vector<int> &associations,
                                     const vector<double> &sense_x,
                                     const vector<double> &sense_y) {
    // particle: the particle to which assign each listed association,
    //   and association's (x,y) world coordinates mapping
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations = associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
    vector<int> v = best.associations;
    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
    vector<double> v;

    if (coord == "X") {
        v = best.sense_x;
    } else {
        v = best.sense_y;
    }

    std::stringstream ss;
    copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length() - 1);  // get rid of the trailing space
    return s;
}

void ParticleFilter::addNoise(Particle &p, double *std) {
    // normal distributions for Gaussian sensor noise
    normal_distribution<double> noise_x(0.0, std[0]);
    normal_distribution<double> noise_y(0.0, std[1]);
    normal_distribution<double> noise_theta(0.0, std[2]);

    // add noise;
    p.x += noise_x(gen);
    p.y += noise_y(gen);
    p.theta += noise_theta(gen);
}

double
ParticleFilter::multivariate_gaussian_probability(double x_map, double y_map, double mu_x, double mu_y, double sigma_x,
                                                  double sigma_y) {

    return 1 / (2 * M_PI * sigma_x * sigma_y) *
           exp(-(pow(x_map - mu_x, 2) / (2 * pow(sigma_x, 2)) + pow(y_map - mu_y, 2) / (2 * pow(sigma_y, 2))));
}
