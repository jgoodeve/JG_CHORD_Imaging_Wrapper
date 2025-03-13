#include <math.h>

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

inline void rotate (const double v [3], double outvec [3], const double alpha)
{
    outvec[0] = cos(alpha)*v[0] - sin(alpha)*v[1];
    outvec[1] = sin(alpha)*v[0] + cos(alpha)*v[1];
    outvec[2] = v[2];
}

inline double dot (const double v1 [3], const double v2 [3])
{
    return v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2];
}

inline double crossmag(const double v1 [3], const double v2 [3])
{
    double cv [3];
    cross(v1,v2,cv);
    return sqrt(dot(cv,cv));
}
