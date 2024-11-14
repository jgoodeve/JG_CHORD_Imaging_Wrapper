//compiling line: nvcc --shared -o dms.so --compiler-options -fPIC dirty_map.cu
#include <cuda_runtime.h>
#include <cuda.h>
#include <math.h>       /* sin, cos, fmod, fabs, asin, atan2 */
#include <stdio.h>
#include <climits>  /*UINT_MAX*/
#include <iostream>

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
    outvec[0] = sin(theta)*cos(phi);
    outvec[1] = sin(theta)*sin(phi);
    outvec[2] = cos(theta);
}

__device__ inline void cross (const float v1 [3], const float v2 [3], float outvec [3])
{
    outvec[0] = v1[1]*v2[2]-v1[2]*v2[1];
    outvec[1] = v1[2]*v2[0]-v1[0]*v2[2];
    outvec[2] = v1[0]*v2[1]-v1[1]*v2[0]; 
}

__device__ inline void rotate (const float v [3], float outvec [3], const float alpha)
{
    outvec[0] = cos(alpha)*v[0] - sin(alpha)*v[1];
    outvec[1] = sin(alpha)*v[0] + cos(alpha)*v[1];
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
    return sqrt(dot(cv,cv));
}

__device__ float B_sq (const float alpha, const float wavelength, const float D)
{
    float alphaprime = PI*D*sin(alpha)/wavelength;
    if (alphaprime <= 1F-8 && alphaprime >= -1F-8)
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
        if (dp < 0.99f) delta_ang = std::acos(dp);    
        else 
        {
            delta_ang = std::asin(crossmag(v1,v2));
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
    float x = fmodf(x_prime,PI); // -pi < x < pi
    x = fabs(x); // 0 < x < pi
    x = (x > PI/2) ? PI-x : x; //0 < x < pi/2
    
    if (fabs(x) < 1E-9) return m*m*cos(m*x)*cos(m*x)/(cos(x)*cos(x));
    else return sin(m*x)*sin(m*x)/(sin(x)*sin(x));
}

__global__ void dirtymap_kernel (const floatArray u, const floatArray wavelengths, const floatArray source_positions, const floatArray source_spectra, 
	float brightness_threshold, const chordParams cp, float * dm, unsigned int pixSegsPerBlock)
{
    //int deviceID;
    //cudaGetDevice(&deviceID);
	//calculating the relevant CHORD vectors for each dither direction
	float * chord_pointing = new float [3*cp.thetas.l];
	float * dir1_proj_vec = new float [3*cp.thetas.l]; //north/south chord direction
	float * dir2_proj_vec = new float [3*cp.thetas.l]; //east/west chord direction
	for (unsigned int k = 0; k < cp.thetas.l; k++)
	{
	    ang2vec(cp.thetas.p[k], 0, chord_pointing+3*k);
	    ang2vec(cp.thetas.p[k] + PI/2, 0, dir1_proj_vec+3*k);
	    cross(dir1_proj_vec+3*k, chord_pointing+3*k, dir2_proj_vec+3*k);
	}
	//accounting for CHORD's baseline shrinking when it points away from zenith
	float * L1s = new float [cp.thetas.l];
	for (unsigned int k = 0; k < cp.thetas.l; k++)
	{
	    L1s[k] = cp.L1*cos(PI/180*(90-cp.CHORD_zenith_dec) - cp.thetas.p[k]);
	}

	for (unsigned int ps = 0; ps < pixSegsPerBlock; ps++)
	{
	    unsigned int pixelIdx = blockIdx.x*32*pixSegsPerBlock + ps*32 + threadIdx.x;
	    if (pixelIdx*3 < u.l)
	    {
	        float * threadu = u.p + pixelIdx*3;
		for (unsigned int l = 0; l < wavelengths.l; l++)
	        {
	            float usum = 0;
	            for (unsigned int s = 0; s*wavelengths.l < source_spectra.l; s++)
	            {
			float time_sum = 0;
	                if (source_spectra.p[s*wavelengths.l + l] > brightness_threshold)
	                {
	     		    float source_phi = atan2(source_positions.p[s*3+1],source_positions.p[s*3]);
			    float initial_travelangle = -source_phi-cp.initial_phi_offset; //we want it to start computing phi_offset away from the source
			    for (unsigned int k = 0; k < cp.thetas.l; k++)
	                    {
	                        for (unsigned int j = 0; j < cp.time_samples; j++)
	                        {
	                            float travelangle = initial_travelangle+j*cp.delta_tau*omega;
	                            float u_rot [3];
	                            rotate(threadu, u_rot, travelangle);
	                            float source_rot [3];
	                            rotate(source_positions.p+3*s, source_rot, travelangle);

	                            float cdir1 = PI*L1s[k]/wavelengths.p[l]*subtractdot(source_rot, u_rot, dir1_proj_vec+3*k);
	                            float cdir2 = PI*cp.L2 /wavelengths.p[l]*subtractdot(source_rot, u_rot, dir2_proj_vec+3*k);

	                            float Bsq_source = Bsq_from_vecs(source_rot, chord_pointing+3*k, wavelengths.p[l], cp.D);
	                            float Bsq_u = Bsq_from_vecs(u_rot, chord_pointing+3*k, wavelengths.p[l], cp.D);

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
        delete chord_pointing;
        delete dir1_proj_vec;
        delete dir2_proj_vec;
        delete L1s;
}

inline void copyFloatArrayToDevice (const floatArray host_array, floatArray & device_array)
{
    device_array.l = host_array.l;

    cudaError_t err = cudaSuccess;
    err = cudaMalloc(&device_array.p, sizeof(float) * host_array.l);
    if (err != cudaSuccess) {fprintf(stderr, "Failed to allocate memory for array (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}

    err = cudaMemcpyAsync(device_array.p, host_array.p, sizeof(float) * host_array.l, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {fprintf(stderr, "Failed to copy data to device array (error code %s)!\n", cudaGetErrorString(err)); exit(EXIT_FAILURE);}
}

unsigned int smallest_remainder (unsigned int min, unsigned int max, unsigned int num)
{
	unsigned int bg = min; //best guess
	unsigned int br = UINT_MAX; //best remainder
	for (unsigned int i = min; i <= max; i++)
	{
		unsigned int quotient = (num+i-1)/i;
		unsigned int remainder = i*quotient - num;
		if (remainder < br)
			{
				bg = i;
				br = remainder;
			}
	}
	return bg;
}

extern "C" {void dirtymap_caller(const floatArray u, const floatArray wavelengths, const floatArray source_positions, const floatArray source_spectra, float brightness_threshold, const chordParams cp, float * dm)
{
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    //std::cout << "Device count: " << deviceCount << std::endl;
    unsigned int npixels = u.l/3;
    if (npixels <= 32) deviceCount = 1; //this is a debugging mode
    //there are 4 GPUs, and each of them cover a quarter of the pixels
    //we're calling a pixSeg (pixel segment) a group of 32 pixels
    unsigned int pixSegsToCover = (npixels+31)/32;
    unsigned int pixSegsPerGPU = (pixSegsToCover+deviceCount-1)/deviceCount;

    float * d_dm [deviceCount]; //array that holds pointers to the deviceCount output arrays
    floatArray d_u[deviceCount];
    floatArray d_wavelengths[deviceCount];
    floatArray d_source_positions[deviceCount];
    floatArray d_source_spectra[deviceCount];
    floatArray d_thetas[deviceCount];

    if (npixels <= 32) deviceCount = 1; //this is a debugmode thing to get it to run on only 1 gpu.
    for (int gpuId = 0; gpuId < deviceCount; gpuId++)
    {
	cudaSetDevice(gpuId);
	//copying data over to the device
        unsigned int npixels_per_gpu = ((gpuId+1) * pixSegsPerGPU * 32 <= npixels) ? pixSegsPerGPU * 32 : npixels - (deviceCount-1) * pixSegsPerGPU * 32;
	floatArray u_for_gpu;
	u_for_gpu.p = u.p + (gpuId * pixSegsPerGPU * 32)*3;
	u_for_gpu.l = npixels_per_gpu*3;
	copyFloatArrayToDevice(u_for_gpu,d_u[gpuId]);
	copyFloatArrayToDevice(wavelengths,d_wavelengths[gpuId]);
	copyFloatArrayToDevice(source_positions, d_source_positions[gpuId]);
	copyFloatArrayToDevice(source_spectra,d_source_spectra[gpuId]);
	copyFloatArrayToDevice(cp.thetas,d_thetas[gpuId]);

        //allocating the return array
        cudaMalloc(&(d_dm[gpuId]), sizeof(float)*npixels_per_gpu*wavelengths.l);
    }

    //launching the kernels on all the GPUs
    //on p100s, I want to launch ~7168 threads, or 224 blocks
    //on V100s, I want to launch ~10240 threads, or 320 blocks
    //let's generalize this to work with any GPU
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int approxBlocksToLaunch = deviceProp.multiProcessorCount * 4;
    unsigned int blocksPerGPU = smallest_remainder(approxBlocksToLaunch-25, approxBlocksToLaunch+25, pixSegsPerGPU);
    for (int gpuId = 0; gpuId < deviceCount; gpuId++)
    {
	cudaSetDevice(gpuId);
	//int deviceId;
	//cudaGetDevice(&deviceId);

	chordParams d_cp = cp;
	d_cp.thetas = d_thetas[gpuId];

	unsigned int pixSegsPerBlock = (pixSegsPerGPU + blocksPerGPU - 1)/blocksPerGPU;
	dirtymap_kernel<<<blocksPerGPU,32>>>(d_u[gpuId], d_wavelengths[gpuId], d_source_positions[gpuId], d_source_spectra[gpuId], brightness_threshold, d_cp, d_dm[gpuId], pixSegsPerBlock);
    }

    cudaDeviceSynchronize();

    cudaError_t kernel_err = cudaGetLastError();
    if (kernel_err != cudaSuccess) std::cout << "Error from kernel: " << kernel_err << std::endl;

    //copying over the data from the GPUs when they're done running
    for (int gpuId = 0; gpuId < deviceCount; gpuId++)
    {
	cudaSetDevice(gpuId);
        unsigned int npixels_per_gpu = ((gpuId+1) * pixSegsPerGPU * 32 <= npixels) ? pixSegsPerGPU * 32 : npixels - (deviceCount-1) * pixSegsPerGPU * 32;
        cudaMemcpyAsync(dm + gpuId * pixSegsPerGPU * 32 * wavelengths.l, d_dm[gpuId], sizeof(float)*npixels_per_gpu*wavelengths.l, cudaMemcpyDeviceToHost);
        cudaFree(d_dm[gpuId]);
	cudaFree(d_u[gpuId].p);
	cudaFree(d_wavelengths[gpuId].p);
	cudaFree(d_source_positions[gpuId].p);
	cudaFree(d_source_spectra[gpuId].p);
	cudaFree(d_thetas[gpuId].p);
    }
}
}

int main ()
{
	size_t free, total;
	cudaFree(0);
	cudaMemGetInfo(&free,&total);
	std::cout << "free: " << free << " total: " << total << std::endl;
}
