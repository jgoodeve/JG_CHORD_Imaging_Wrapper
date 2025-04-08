//compile line: nvcc --shared -o gpu_dmns.so --compiler-options -fPIC dm_noise_sim.cu
#include <curand_kernel.h>
#include <math.h> //sqrt
#include <complex>

#define PI 3.14159265358979323846
#define PIXELS_PER_BLOCK 128

typedef struct
{
	float x;
	float y;
	float z;
} vector3;

__device__ void fill_noise_draws (float* noise_draws, int n, float noise, unsigned long long seed)
{
	int idx = threadIdx.x + blockIdx.x *32;
	curandStateMRG32k3a_t state;
	curand_init(seed, idx, 0, &state);
	if (idx < n)
	{
		noise_draws[idx] = curand_normal(&state);
	}
}

inline void ang2vec (const double theta, const double phi, double outvec [3])
{
    outvec[0] = sin(theta)*cos(phi);
    outvec[1] = sin(theta)*sin(phi);
    outvec[2] = cos(theta);
}

inline void cross (const double v1 [3], const double v2 [3], double outvec [3])
{
    outvec[0] = v1[1]*v2[2]-v1[2]*v2[1];
    outvec[1] = v1[2]*v2[0]-v1[0]*v2[2];
    outvec[2] = v1[0]*v2[1]-v1[1]*v2[0]; 
}

__device__ inline void rotate (const float v [3], float outvec [3], const float alpha)
{
    outvec[0] = cosf(alpha)*v[0] - sinf(alpha)*v[1];
    outvec[1] = sinf(alpha)*v[0] + cosf(alpha)*v[1];
    outvec[2] = v[2];
}

__device__ inline float dot (const float v1 [3], const float v2 [3])
{
    return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2];
}

__device__ float B_sq (const float alpha, const float wavelength, const float D)
{
    float alphaprime = PI*D*sinf(alpha)/wavelength;
    if (alphaprime <= 1E-8f && alphaprime >= -1E-8f)
        return (j0f(alphaprime)-jnf(2,alphaprime))*(j0f(alphaprime)-jnf(2,alphaprime)); //l'Hopital's
    else
        return (2*j1f(alphaprime)/alphaprime) * (2*j1f(alphaprime)/alphaprime);
}

__device__ inline float Bsq_from_vecs (const float v1 [3], const float v2 [3], const float wavelength, const float D)
{
    float dp = dot(v1,v2);
    if (dp <= 0) return 0; //horizon condition
    else
    {
        //we want to deal with the arccos instiblity by using the cross product formula instead
        float delta_ang;
        if (dp < 0.99f) delta_ang = acosf(dp);
        else
        {
            delta_ang = asinf(crossmag(v1,v2));
            //delta_ang = (dp > 0) ? delta_ang : PI-delta_ang; //I don't need this line with the horizon condition
        }
        return B_sq(delta_ang, wavelength, D);
    }
}

__device__ void dm_noise_sim (const std::complex<float>* visibility_noise_draws, float* inv_cov,
    const float* u, const float* baselines, int nbaselines, float* wavelength,
    vector3 telescope_u, float dish_diameter, float initial_phi_offset, int ntimesamples, float* noise_map)
{
    int pixelidx = threadIdx.x + blockIdx.x*PIXELS_PER_BLOCK;
    for (int l = blockIdx.y*nwavelenths_per_block; l<blockIdx.y*nwavelenths_per_block+32; l++)
    {
        std::complex<float> sum = 0;
        for (int i = 0; i < nbaselines; i++)
        {
            for (int t = 0; t < ntimesamples; t++)
            {
                float phi = -initial_phi_offset + (2*initial_phi_offset/ntimesamples)*t;
                float u_rot [3];
                rotate(u[3*pixelidx], u_rot, phi);
                float Bsq = Bsq_from_vecs (u_rot, telescope_u, wavelengths[l], dish_diamater);
                sum += inv_cov[i]*visibility_noise_draws[i*ntimesamples+t]*exp(-2*PI*1i/wavelengths[l]*dot(baselines[i*3],u_rot)); //this doesn't work because the noise draws won't match between us
            }
        }
        noise_map[pixelidx*nwavelengths+l] = sum.real() + sum.imag();
    }
}

extern "C" {void dm_noise_sim_caller (float noise,
    const float* u, unsigned int npixels, const float* baselines, const int* baseline_counts, int nbaselines, const float* wavelengths, int nwavelengths,
    float telescope_dec, float dish_diameter, float deg_distance_to_count, int ntimesamples_full, float* noise_map, unsigned long long seed)
{   
	using namespace std::complex_literals;
	
    float telescope_theta = (90-telescope_dec)*PI/180;

    //making unit vector baselines
    float zenith_basis [3];
    float ns_basis [3];
    float ew_basis [3];
    ang2vec(telescope_theta,0,zenith_basis);
    ang2vec(telescope_theta-PI/2,0,ns_basis);
    cross(ns_basis,zenith_basis,ew_basis);

    float* a = new float [3*nbaselines];
    for (int i = 0; i < nbaselines; i++)
    {
        a[3*i+0] = baselines[2*i]*ew_basis[0] + baselines[2*i+1]*ns_basis[0];
        a[3*i+1] = baselines[2*i]*ew_basis[1] + baselines[2*i+1]*ns_basis[1];
        a[3*i+2] = baselines[2*i]*ew_basis[2] + baselines[2*i+1]*ns_basis[2];
    }

	//setting up multiple GPUs and cuda stuff
	int deviceCount;
    cudaGetDeviceCount(&deviceCount);
	std::cout << "Device count: " << deviceCount << std::endl;
	
	//padding
	unsigned int npixelblocks_padded = (npixels+PIXELS_PER_BLOCK-1)/PIXELS_PER_BLOCK;
	int npixels_padded = npixelblocks_padded*PIXELS_PER_BLOCK;
	unsigned int gpu_pixel_idx[deviceCount]; //where each GPU starts considering pixels
	unsigned int gpu_pixel_length[deviceCount]; //how many pixels the GPU computes
	for (int gpuId = 0; gpuId < deviceCount; gpuId++)
	{
		unsigned int pixelblock_idx = npixelblocks_padded/deviceCount * gpuId;
		gpu_pixel_idx[gpuId] = pixelblock_idx * PIXELS_PER_BLOCK;
		unsigned int pixelblock_length = gpuId + 1 == deviceCount ? npixelblocks_padded - pixelblock_idx : npixelblocks_padded/deviceCount;
		gpu_pixel_length[gpuId] = pixelblock_length * PIXELS_PER_BLOCK;
	}
	float* u_padded = new float [npixels_padded*3];
	for (unsigned int i = 0; i<npixels*3; i++) u_padded[i] = u[i];
	for (unsigned int i = npixels; i < npixels_padded; i++)
    {
		u_padded[3*i]   = 1;
		u_padded[3*i+1] = 0;
		u_padded[3*i+2] = 0;
    }
    float* u_d[deviceCount];
    
	float* padded_noise_map_d[deviceCount];
	float* noise_draws[deviceCount];
	float* a_d[deviceCount];
	for (int gpuId = 0; gpuId < deviceCount; gpuId++)
	{
		cudaSetDevice(gpuId);
		cudaMalloc(&u_d[gpuId], sizeof(float)*gpu_pixel_length[gpuId]*3);
		cudaMemcpyAsync(d_u[gpuId], u + gpu_pixel_idx[gpuId]*3, sizeof(float)*gpu_pixel_length[gpuId]*3, cudaMemcpyHostToDevice);
		cudaMalloc(&padded_noise_map_d[gpuId], sizeof(float)*gpu_pixel_length[gpuId]);
		cudaMalloc(&noise_draws[gpuId], sizeof(float)*nbaselines*2);
		cudaMalloc(&a_d[gpuId], sizeof(float)*nbaselines*3);
		cudaMemcpyAsync(baselines_d[gpuId], a, sizeof(float)*nbaselines*3, cudaMemcpyHostToDevice);
	}
	delete[] u_padded;
	delete[] a;
	for (int l = 0; l < nwavelengths; l++)
	{
		
	}

	for (int gpuId = 0; gpuId < deviceCount; gpuId++)
	{
		cudaSetDevice(gpuId);
		cudaFree(u_d[gpuId]);
		cudaFree(padded_noise_map_d[gpuId]);
		cudaFree(noise_draws[gpuId]);
		cudaFree(a_d[gpuId]);
	}
	
    //making random visibility noise draws
    std::default_random_engine rng;

    std::complex<float>* noise_draws = new std::complex<float>[nbaselines*ntimesamples_full*nwavelengths];
    float* inv_cov = new float [nbaselines];
    for (int i = 0; i < nbaselines; i++)
    {
        float stdv;
        if (baselines[2*i] == 0 && baselines[2*i+1] == 0)
        {
            stdv = sqrt(baseline_counts[i])*noise * sqrt(2)/2;
        }
        else
            stdv = sqrt(baseline_counts[i])*noise;
        inv_cov[i] = 1/(stdv*stdv);
        std::normal_distribution<float> noise_distribution (0, stdv);
        for (int l = 0; l < nwavelengths; l++)
        {
            for (int j = 0; j <  ntimesamples; j++)
            {
                noise_draws[l*nbaselines*ntimesamples + i*ntimesamples+j] = noise_distribution(rng) + noise_distribution(rng)*1i;
            }
        }
    }
    
}
}
