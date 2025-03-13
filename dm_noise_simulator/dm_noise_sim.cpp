//compiling line
//g++ -fPIC -o dmns.so -shared dm_noise_sim.cpp -fopenmp

#include <iostream>
#include <random>
#include <math.h> //sqrt
#include <complex>
#include <omp.h>
#include "vector_util.h"

#define PI 3.14159265358979323846

double B_sq (const double alpha, const double wavelength, double D)
{
    double alphaprime = PI*D*sin(alpha)/wavelength;
    if (alphaprime <= 1E-8 && alphaprime >= -1E-8)
        return (std::cyl_bessel_j(0,alphaprime)-std::cyl_bessel_j(2,alphaprime))*(std::cyl_bessel_j(0,alphaprime)-std::cyl_bessel_j(2,alphaprime)); //l'Hopital's
    else
        return (2*std::cyl_bessel_j(1,alphaprime)/alphaprime) * (2*std::cyl_bessel_j(1,alphaprime)/alphaprime);
}

inline double Bsq_from_vecs (const double v1 [3], const double v2 [3], const double wavelength, double D)
{
    double dp = dot(v1,v2);
    if (dp <= 0) return 0; //horizon condition
    else
    {
        //we want to deal with the arccos instiblity by using the cross product formula instead
        double delta_ang;
        if (dp < 0.99) delta_ang = std::acos(dp);    
        else 
        {
            delta_ang = std::asin(crossmag(v1,v2));
            //delta_ang = (dp > 0) ? delta_ang : PI-delta_ang; //I don't need this line with the horizon condition
        }
        return B_sq(delta_ang, wavelength, D);
    }
}

extern "C" {void dm_noise_sim (double noise,
    const double* u, int npixels, const double* baselines, const int* baseline_counts, int nbaselines, const double* wavelengths, int nwavelengths,
    double telescope_dec, double dish_diameter, double deg_distance_to_count, int ntimesamples_full, double* noise_map)
{   
	using namespace std::complex_literals;
	
    double telescope_theta = (90-telescope_dec)*PI/180;

    //making unit vector baselines
    double zenith_basis [3];
    double ns_basis [3];
    double ew_basis [3];
    ang2vec(telescope_theta,0,zenith_basis);
    ang2vec(telescope_theta-PI/2,0,ns_basis);
    cross(ns_basis,zenith_basis,ew_basis);

    double* a = new double [3*nbaselines];
    for (int i = 0; i < nbaselines; i++)
    {
        a[3*i+0] = baselines[2*i]*ew_basis[0] + baselines[2*i+1]*ns_basis[0];
        a[3*i+1] = baselines[2*i]*ew_basis[1] + baselines[2*i+1]*ns_basis[1];
        a[3*i+2] = baselines[2*i]*ew_basis[2] + baselines[2*i+1]*ns_basis[2];
    }

    //making random visibility noise draws
    std::default_random_engine rng;

    std::complex<double>* noise_draws = new std::complex<double>[nbaselines*ntimesamples_full*nwavelengths];
    double* inv_cov = new double [nbaselines];
    for (int i = 0; i < nbaselines; i++)
    {
        double stdv;
        if (baselines[2*i] == 0 && baselines[2*i+1] == 0)
        {
            stdv = sqrt(baseline_counts[i])*noise * sqrt(2)/2;
        }
        else
            stdv = sqrt(baseline_counts[i])*noise;
        inv_cov[i] = 1/(stdv*stdv);
        std::normal_distribution<double> noise_distribution (0, stdv);
        for (int l = 0; l < nwavelengths; l++)
        {
            for (int j = 0; j <  ntimesamples_full; j++)
            {
                noise_draws[l*nbaselines*ntimesamples_full + i*ntimesamples_full+j] = noise_distribution(rng) + noise_distribution(rng)*1i;
            }
        }
    }
    delete[] a;
    
    //we don't want to actually count every time step, since the majority of them are 0 because of B_sq
    int ntimesamples = int(ntimesamples_full * (2*deg_distance_to_count)/360.0);

    std::cout << "Running noise simulator with " << omp_get_max_threads() << " threads." << std::endl;
    #pragma omp parallel for
    for (int j = 0; j < npixels; j++)
    {
        double pixelphi = atan2(*(u+3*j),*(u+3*j+1));
        int rough_time_placement = pixelphi/(2*PI) * ntimesamples_full;
        int t_initial = rough_time_placement-ntimesamples/2;
        for (int l = 0; l<nwavelengths; l++)
        {
        std::complex<double> sum = 0;
		for (int i = 0; i < nbaselines; i++)
        {
            for (int t = t_initial; t < t_initial+ntimesamples; t++)
            {
                double delta_phi = (2*PI/ntimesamples_full)*t;
                double u_rot [3];
                rotate(u+3*j, u_rot, delta_phi);
                double Bsq = Bsq_from_vecs (u_rot, zenith_basis, wavelengths[l], dish_diameter);
                sum += inv_cov[i]*noise_draws[i*ntimesamples_full+t%ntimesamples_full]*Bsq*exp(-2*PI*1i/wavelengths[l]*dot(baselines+i*3,u_rot));
            }
        }
        noise_map[j*nwavelengths+l] = sum.real() + sum.imag();
        }
    }
}
}
