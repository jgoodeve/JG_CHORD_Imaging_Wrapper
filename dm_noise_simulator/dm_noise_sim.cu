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

__device__ void fill_noise_draws (float* noise_draws, int ndraws, int ntimesamples, float* stdv, unsigned long long seed)
{
	int idx = threadIdx.x + blockIdx.x *32;
	curandStateMRG32k3a_t state;
	curand_init(seed, idx, 0, &state);
	if (idx < ndraws)
	{
		noise_draws[idx] = curand_normal(&state)*stdv[idx/(ntimesamples*2)];
	}
}

inline vector3 ang2vec (const double theta, const double phi)
{
	return {sin(theta)*cos(phi), sin(theta)*sin(phi), cos(theta)};
}

inline vector3 cross (const vector3 v1, const vector3 v2)
{
	return {v1.y*v2.z-v1.z*v2.y,
			v1.z*v2.x-v1.x*v2.z,
			v1.x*v2.y-v1.y*v2.x};
}

__device__ inline vector3 rotate (const vector3 v, const float alpha)
{
	return {cosf(alpha)*v.x - sinf(alpha)*v.y, sinf(alpha)*v.x + cosf(alpha)*v.y, v.z};
}

__device__ inline float dot (const vector3 v1, const vector3 v2)
{
    return v1.x*v2.x+v1.y*v2.y+v1.z*v2.z;
}

__device__ inline float angular_difference (const vector3 v1, const vector3 v2)
{
	vector3 c = cross(v1, v2);
	return atan2f(sqrtf(dot(c,c)), dot(v1,v2));
}

__device__ float B_sq (const float alpha, const float wavelength, const float D)
{
    float alphaprime = PI*D*sinf(alpha)/wavelength;
    if (alphaprime <= 1E-8f && alphaprime >= -1E-8f)
        return (j0f(alphaprime)-jnf(2,alphaprime))*(j0f(alphaprime)-jnf(2,alphaprime)); //l'Hopital's
    else
        return (2*j1f(alphaprime)/alphaprime) * (2*j1f(alphaprime)/alphaprime);
}

__device__ void dm_noise_sim (const std::complex<float>* visibility_noise_draws, float* inv_cov,
    const vector3* u, const vector3* baselines, int nbaselines, float* wavelength,
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
                vector3 u_rot = rotate(u[pixelidx], phi);
                float Bsq = B_sq(angular_difference(u_rot, telescope_u), wavelengths[l], dish_diamater);
                sum += inv_cov[i]*visibility_noise_draws[i*ntimesamples+t]*exp(-2*PI*1i/wavelengths[l]*dot(baselines[i],u_rot));
            }
        }
        noise_map[pixelidx*nwavelengths+l] = sum.real();
    }
}

extern "C" {void dm_noise_sim_caller (float noise,
    const vector3* u, unsigned int npixels, const float* baselines, const int* baseline_counts, int nbaselines, const float* wavelengths, int nwavelengths,
    float telescope_dec, float dish_diameter, float deg_distance_to_count, int ntimesamples_full, float* noise_map, unsigned long long seed)
{   
	using namespace std::complex_literals;
	
    float telescope_theta = (90-telescope_dec)*PI/180;

    //making unit vector baselines
    vector3 zenith_basis = ang2vec(telescope_theta,0);
    vector3 ns_basis = ang2vec(telescope_theta-PI/2,0);
    vector3 ew_basis = cross(ns_basis,zenith_basis);

    vector3* a = new vector3 [nbaselines];
    float* stdv = new float [nbaselines]; //setting up list of standard deviations for each baseline
    for (int i = 0; i < nbaselines; i++)
    {
        a[i] = {baselines[2*i]*ew_basis[0] + baselines[2*i+1]*ns_basis[0],
         		baselines[2*i]*ew_basis[1] + baselines[2*i+1]*ns_basis[1],
         		baselines[2*i]*ew_basis[2] + baselines[2*i+1]*ns_basis[2]};
	    if (baselines[2*i] == 0 && baselines[2*i+1] == 0)
	    {
	        stdv[i] = sqrt(baseline_counts[i])*noise * sqrt(2)/2;
	    }
	    else
	        stdv[i] = sqrt(baseline_counts[i])*noise;
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
	vector3* u_padded = new vector3 [npixels_padded];
	for (unsigned int i = 0; i<npixels; i++) u_padded = u[i];
	for (unsigned int i = npixels; i < npixels_padded; i++)
    {
    	u_padded[i] = {1,0,0};
    }
    vector3* u_d[deviceCount];
    
	float* padded_noise_map_d[deviceCount];
	float* noise_draws[deviceCount];
	float* a_d[deviceCount];
	float* stdv_d[deviceCount];
	for (int gpuId = 0; gpuId < deviceCount; gpuId++)
	{
		cudaSetDevice(gpuId);
		cudaMalloc(&u_d[gpuId], sizeof(vector3)*gpu_pixel_length[gpuId]);
		cudaMemcpyAsync(d_u[gpuId], u + gpu_pixel_idx[gpuId], sizeof(vector3)*gpu_pixel_length[gpuId], cudaMemcpyHostToDevice);
		cudaMalloc(&padded_noise_map_d[gpuId], sizeof(float)*gpu_pixel_length[gpuId]);
		cudaMalloc(&noise_draws[gpuId], sizeof(float)*nbaselines*ntimesamples_full*2);
		cudaMalloc(&a_d[gpuId], sizeof(vector3)*nbaselines);
		cudaMemcpyAsync(baselines_d[gpuId], a, sizeof(vector3)*nbaselines, cudaMemcpyHostToDevice);
		cudaMalloc(&stdv_d[gpuId], sizeof(float)*nbaselines);
		cudaMemcpyAsync(stdv_d[gpuId], stdv, sizeof(float)*nbaselines, cudaMemcpyHostToDevice);
	}
	delete[] u_padded;
	delete[] a;
	delete[] stdv;
	for (int l = 0; l < nwavelengths; l++)
	{	
		for (int gpuId = 0; gpuId < deviceCount; gpuId++)
		{
			cudaSetDevice(gpuId);
			int nblocks = (2*nbaselines*ntimesamples_full + 31)/32;
			fill_noise_draws <<<nblocks, 32>>> (noise_draws[gpuId], 2*nbaselines*ntimesamples_full, noise, seed);
		}
		cudaDeviceSynchronize();
		for (int gpuId = 0; gpuId < deviceCount; gpuId++)
		{
			cudaSetDevice(gpuId);
			int nblocks = gpu_pixel_length[gpuId]/PIXELS_PER_BLOCK;
			dm_noise_sim<<<nblocks,PIXELS_PER_BLOCK>>> (static_cast<std::complex<float>>(noise_draws[gpuId]), float* inv_cov,
    			const vector3* u, const vector3* baselines, int nbaselines, float* wavelength,
    			vector3 telescope_u, float dish_diameter, float initial_phi_offset, int ntimesamples, float* noise_map);
		}
	}

	for (int gpuId = 0; gpuId < deviceCount; gpuId++)
	{
		cudaSetDevice(gpuId);
		cudaFree(u_d[gpuId]);
		cudaFree(padded_noise_map_d[gpuId]);
		cudaFree(noise_draws[gpuId]);
		cudaFree(a_d[gpuId]);
	}
}
}
