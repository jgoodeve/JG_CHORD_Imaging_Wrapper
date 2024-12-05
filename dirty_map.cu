//compiling line: nvcc --shared -o dms.so --compiler-options -fPIC dirty_map.cu
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>       /* sin, cos, fmod, fabs, asin, atan2 */
#include <stdio.h>
#include <climits>  /*UINT_MAX*/
#include <iostream>
#include <cassert> /* assert */

#define MAX_DITHERS 5
#define THREADS_PER_BLOCK 128

constexpr float PI = 3.14159265358979323846f;
constexpr float omega = float(2*PI/86400); //earth angular velocity in rads/second

struct floatArray {
    float * p;
    unsigned int l;
};

struct chordParams
{
    floatArray thetas;
    float initial_phi_offset; //amount that the calculation starts away from each source
    unsigned int m1; //north south number of dishes
    unsigned int m2; //east west
    float L1; // north osuth dish separation
    float L2; //east west
    float CHORD_zenith_dec;
    float D; //dish diameter
    float delta_tau;
    unsigned int time_samples;
};

__device__ inline void ang2vec (const float theta, const float phi, float outvec [3])
{
    outvec[0] = sinf(theta)*cosf(phi);
    outvec[1] = sinf(theta)*sinf(phi);
    outvec[2] = cosf(theta);
}

__device__ inline void cross (const float v1 [3], const float v2 [3], float outvec [3])
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

__device__ inline float crossmag(const float v1 [3], const float v2 [3])
{
    float cv [3];
    cross(v1,v2,cv);
    return sqrtf(dot(cv,cv));
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

__device__ inline float subtractdot (const float v1_a [3], const float v1_b [3], const float v2 [3])
{
    return (v1_a[0]-v1_b[0])*v2[0]+(v1_a[1]-v1_b[1])*v2[1]+(v1_a[2]-v1_b[2])*v2[2];
}

__device__ float sin_sq_ratio (const unsigned int m, const float x_prime)
{
    float x = fmodf(x_prime,1.0); // -1.0 < x < 1.0
    x = fabs(x); // 0 < x < 1.0
    x = (x > 0.5f) ? 1-x : x; //0 < x < 0.5
    
    if (fabs(x) < 1E-9f) return m*m*cospif(m*x)*cospif(m*x)/(cospif(x)*cospif(x));
    else return sinpif(m*x)*sinpif(m*x)/(sinpif(x)*sinpif(x));
}

__global__ void precompute (float * precomputed_array, const chordParams cp)
{
	//it should be organized like dither 1, 3 vectors and then the L1, dither 2 3 vectors and L1 and so on
	for (unsigned int i = 0; i<MAX_DITHERS && i<cp.thetas.l; i++)
	{
		float * chord_pointing = precomputed_array + 10*i; //calculating baseline unit vectors
		float * dir1_proj_vec  = precomputed_array + 10*i+3;
		float * dir2_proj_vec  = precomputed_array + 10*i+6;
	        float * L1_modified = precomputed_array+10*i+9; //accounting for CHORD's baseline shrinking when it points away from zenith
		ang2vec(cp.thetas.p[i], 0, chord_pointing);
	        ang2vec(cp.thetas.p[i] + PI/2, 0, dir1_proj_vec);
   	        cross(dir1_proj_vec, chord_pointing, dir2_proj_vec);
		*L1_modified = cp.L1*cosf(PI/180*(90-cp.CHORD_zenith_dec) - cp.thetas.p[i]);
	}
}

__global__ void dirtymap_kernel (const floatArray u, const floatArray wavelengths, const floatArray source_positions, const floatArray source_spectra, 
	float brightness_threshold, const chordParams cp, float * dm, const float precompute_array[10*MAX_DITHERS])
{
    //int deviceID;
    //cudaGetDevice(&deviceID);
    unsigned int pixelIdx = blockIdx.x*THREADS_PER_BLOCK + threadIdx.x;
    if (pixelIdx*3 < u.l)
    {
        float * threadu = u.p + pixelIdx*3;
	for (unsigned int l = blockIdx.y*32; l < blockIdx.y*32+32; l++)
        {
            float usum = 0;
            for (unsigned int s = 0; s*wavelengths.l < source_spectra.l; s++)
            {
		float time_sum = 0;
                if (source_spectra.p[s*wavelengths.l + l] > brightness_threshold)
                {
     		    float source_phi = atan2f(source_positions.p[s*3+1],source_positions.p[s*3]);
		    float initial_travelangle = -source_phi-cp.initial_phi_offset; //we want it to start computing phi_offset away from the source
		    for (unsigned int k = 0; k < cp.thetas.l && k < MAX_DITHERS; k++)
                    {
			const float * chord_pointing = precompute_array + 10*k;
			const float * dir1_proj_vec  = precompute_array + 10*k+3;
			const float * dir2_proj_vec  = precompute_array + 10*k+6;
			const float L1_modified = *(precompute_array+10*k+9);
                        for (unsigned int j = 0; j < cp.time_samples; j++)
                        {
                            float travelangle = initial_travelangle+j*cp.delta_tau*omega;
                            float u_rot [3];
                            rotate(threadu, u_rot, travelangle);
                            float source_rot [3];
                            rotate(source_positions.p+3*s, source_rot, travelangle);

                            float cdir1 = L1_modified/wavelengths.p[l]*subtractdot(source_rot, u_rot, dir1_proj_vec);
                            float cdir2 = cp.L2/wavelengths.p[l]*subtractdot(source_rot, u_rot, dir2_proj_vec);

                            float Bsq_source = Bsq_from_vecs(source_rot, chord_pointing, wavelengths.p[l], cp.D);
                            float Bsq_u = Bsq_from_vecs(u_rot, chord_pointing, wavelengths.p[l], cp.D);

                            time_sum += Bsq_source * Bsq_u * sin_sq_ratio(cp.m1,cdir1) * sin_sq_ratio(cp.m2,cdir2);
                        }
                    }
                }
                usum += source_spectra.p[s*wavelengths.l + l] * time_sum;
            }
            dm[pixelIdx*wavelengths.l + l] += usum;
        }
    }
}

/*__global__ void transpose ()
{
}*/

inline void copyFloatArrayToDevice (const floatArray host_array, floatArray & device_array)
{
    device_array.l = host_array.l;

    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&device_array.p, sizeof(float) * host_array.l);
    if (err != cudaSuccess) {fprintf(stderr, "Failed to allocate memory for array (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}

    err = cudaMemcpyAsync(device_array.p, host_array.p, sizeof(float) * host_array.l, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {fprintf(stderr, "Failed to copy data to device array (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
}

extern "C" {void dirtymap_caller(const floatArray u, const floatArray wavelengths, const floatArray source_positions, const floatArray source_spectra, float brightness_threshold, const chordParams cp, float * dm)
{
    assert(cp.thetas.l <= MAX_DITHERS);

    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    //std::cout << "Device count: " << deviceCount << std::endl;
    unsigned int npixels = u.l/3;
    if (npixels <= THREADS_PER_BLOCK) deviceCount = 1; //this is a debugging mode
    //there are 4 GPUs, and each of them cover a quarter of the pixels
    //let's calculate how many pixels we need so that this divides evenly
    unsigned int npixels_padded = (npixels + (deviceCount * THREADS_PER_BLOCK) - 1)/(deviceCount * THREADS_PER_BLOCK) * (deviceCount * THREADS_PER_BLOCK); //this seems weird but works
    unsigned int nblocks_per_gpu = npixels_padded/(deviceCount*THREADS_PER_BLOCK); //this is kind of a misnomer. This means number of blocks in the x dimension. Over freq dimension is different.
    unsigned int npixels_per_gpu = nblocks_per_gpu * THREADS_PER_BLOCK;
    unsigned int nwavelengths_padded = (wavelengths.l+31)/32 * 32; // padded out to be a multiple of 32
    unsigned int nsources = source_spectra.l/wavelengths.l;
    dim3 nblocks_both_dims (nblocks_per_gpu, nwavelengths_padded/32); //we want to call one block per 32 wavelengths

    //padding
    float * u_padding = new float [(npixels_padded-npixels)*3];
    for (unsigned int i = 0; i < npixels_padded-npixels; i++)
    {
	u_padding[3*i]   = 1;
	u_padding[3*i+1] = 0;
	u_padding[3*i+2] = 0;
    }
    floatArray wavelengths_padded;
    wavelengths_padded.p = new float [nwavelengths_padded];
    wavelengths_padded.l = nwavelengths_padded;
    for (unsigned int i = 0; i < wavelengths.l; i++) wavelengths_padded.p[i] = wavelengths.p[i];
    for (unsigned int i = wavelengths.l; i < nwavelengths_padded; i++) wavelengths_padded.p[i] = 1E10f; //some crazy number to help tell that it's fake in case I copy the wrong thing.
    floatArray source_spectra_padded;
    source_spectra_padded.p = new float [nwavelengths_padded*nsources];
    source_spectra_padded.l = nwavelengths_padded*nsources;
    for (unsigned int j = 0; j < nsources; j++)
    {
	for (unsigned int i = 0; i < wavelengths.l; i++) source_spectra_padded.p[j*nwavelengths_padded+i] = source_spectra.p[j*wavelengths.l+i];
        for (unsigned int i = wavelengths.l; i < nwavelengths_padded; i++) source_spectra_padded.p[j*nwavelengths_padded+i] = 0;
    }
    //padding over

    float * d_dm [deviceCount]; //array that holds pointers to the deviceCount output arrays
    floatArray d_u[deviceCount];
    floatArray d_wavelengths[deviceCount];
    floatArray d_source_positions[deviceCount];
    floatArray d_source_spectra[deviceCount];
    floatArray d_thetas[deviceCount];

    float * precompute_array[deviceCount];

    for (int gpuId = 0; gpuId < deviceCount; gpuId++)
    {
	cudaSetDevice(gpuId);
	//copying data over to the device
	copyFloatArrayToDevice(wavelengths_padded,d_wavelengths[gpuId]);
	copyFloatArrayToDevice(source_positions, d_source_positions[gpuId]);
	copyFloatArrayToDevice(source_spectra_padded,d_source_spectra[gpuId]);
	copyFloatArrayToDevice(cp.thetas,d_thetas[gpuId]);

	//copying over the u vectors that we need
        cudaMalloc(&d_u[gpuId].p, sizeof(float) * npixels_per_gpu * 3);
	unsigned int pixels_to_copy = (gpuId+1) * npixels_per_gpu < npixels ? npixels_per_gpu : npixels - gpuId*npixels_per_gpu;
	cudaMemcpyAsync(d_u[gpuId].p, u.p + gpuId*npixels_per_gpu*3, sizeof(float)*pixels_to_copy*3, cudaMemcpyHostToDevice);
	cudaMemcpyAsync(d_u[gpuId].p, u_padding, sizeof(float) * (npixels_per_gpu-pixels_to_copy)*3, cudaMemcpyHostToDevice); //copying over the padding

	//allocating the precompute array
	cudaMalloc(&(precompute_array[gpuId]), sizeof(float)*10*MAX_DITHERS);
        //allocating the return array
        cudaMalloc(&(d_dm[gpuId]), sizeof(float)*npixels_per_gpu*wavelengths_padded.l);
    }
    cudaDeviceSynchronize();
    delete u_padding; //we don't need this anymore I think, making sure we sync before deleting
    delete wavelengths_padded.p;
    delete source_spectra_padded.p;

    cudaError_t copying_err = cudaGetLastError();
    if (copying_err != cudaSuccess) std::cout << "Error from copying: " << copying_err << std::endl;

    //running a quick kernel to precompute some values on all the GPUs
    for (int gpuId = 0; gpuId < deviceCount; gpuId++)
    {
        cudaSetDevice(gpuId);
	chordParams d_cp = cp;
	d_cp.thetas = d_thetas[gpuId];
	precompute<<<1,1>>>(precompute_array[gpuId], d_cp);
    }

    //launching main kernel
    for (int gpuId = 0; gpuId < deviceCount; gpuId++)
    {
	cudaSetDevice(gpuId);

	chordParams d_cp = cp;
	d_cp.thetas = d_thetas[gpuId];

	dirtymap_kernel<<<nblocks_both_dims,THREADS_PER_BLOCK>>>(d_u[gpuId], d_wavelengths[gpuId], d_source_positions[gpuId], d_source_spectra[gpuId], brightness_threshold, d_cp, d_dm[gpuId],
		precompute_array[gpuId]);
    }

    cudaDeviceSynchronize();

    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) std::cout << "Error from kernel: " << kernel_err << std::endl;

    //copying over the data from the GPUs when they're done running
    float * dm_padded = new float[npixels_padded*wavelengths_padded.l];
    for (int gpuId = 0; gpuId < deviceCount; gpuId++)
    {
	cudaSetDevice(gpuId);
        cudaMemcpyAsync(dm_padded + gpuId * npixels_per_gpu * wavelengths_padded.l, d_dm[gpuId], sizeof(float)*npixels_per_gpu*wavelengths_padded.l, cudaMemcpyDeviceToHost);
        cudaFree(d_dm[gpuId]);
	cudaFree(d_u[gpuId].p);
	cudaFree(d_wavelengths[gpuId].p);
	cudaFree(d_source_positions[gpuId].p);
	cudaFree(d_source_spectra[gpuId].p);
	cudaFree(d_thetas[gpuId].p);
	cudaFree(precompute_array[gpuId]);
    }
    cudaDeviceSynchronize();

    //now we need to de-pad
    for (unsigned int i = 0; i < npixels; i++)
    {
	for (unsigned int j = 0; j < wavelengths.l; j++) dm[i*wavelengths.l+j] = dm_padded[i*wavelengths_padded.l+j];
    }
    delete dm_padded;
}
}

int main ()
{
	size_t free, total;
	cudaFree(0);
	cudaMemGetInfo(&free,&total);
	std::cout << "free: " << free << " total: " << total << std::endl;
}
