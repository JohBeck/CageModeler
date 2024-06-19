#ifndef CPU_IMPL_H
#define CPU_IMPL_H

// #include <cuda_runtime.h>
// #include <helper_cuda.h>
#include <iostream>
#include <cstring>

typedef float scalar_t;
typedef int   index_t;

extern "C" {
  
  void mvc_gpu(scalar_t *d_PHI,
               const scalar_t *d_V,
               const index_t  *d_cageF,
               const scalar_t *d_cageV,               
               const index_t nv,
               const index_t ncf,
               const index_t ncv);

  void green_gpu(scalar_t *d_phix,
                 scalar_t *d_phiy,
                 scalar_t *d_phiz,
                 scalar_t *d_psi,
                 const scalar_t *d_V,
                 const index_t  *d_cageF,
                 const scalar_t *d_cageV,
                 const scalar_t *d_cageN,
                 const index_t nv,
                 const index_t ncf,
                 const index_t ncv,
                 const scalar_t *d_qp,
                 const scalar_t *d_qw,
                 const index_t nq);

  void green_gpu_post(scalar_t *d_phi,
                      const scalar_t *d_phix,
                      const scalar_t *d_phiy,
                      const scalar_t *d_phiz,
                      const index_t  *d_cageF,
                      const index_t nv,
                      const index_t ncf,
                      const index_t ncv);  

  void somig_gpu(const scalar_t nu,
                 scalar_t *d_PHIx,
                 scalar_t *d_PHIy,
                 scalar_t *d_PHIz,
                 scalar_t *d_PSI ,
                 const scalar_t *d_V,
                 const index_t  *d_cageF,
                 const scalar_t *d_cageV,
                 const scalar_t *d_cageN,
                 const index_t nv,
                 const index_t ncf,
                 const index_t ncv,
                 const scalar_t *d_qp,
                 const scalar_t *d_qw,
                 const index_t nq);

  // reduce PHIxyz to PHI
  void somig_gpu_post(scalar_t *d_PHI,
                      const scalar_t *d_PHIx,
                      const scalar_t *d_PHIy,
                      const scalar_t *d_PHIz,
                      const index_t  *d_cageF,
                      const index_t nv,
                      const index_t ncf);
  
}

class cuda_cage_precomputer
{
 public:
 ~cuda_cage_precomputer() {
  // mesh and points
  delete[] d_cageF_;
  delete[] d_cageV_;
  delete[] d_cageN_;
  delete[] d_V_;

  // quadratures
  delete[] d_qp_;
  delete[] d_qw_;
  
    // // green
    // cudaFree(d_phix_);
    // cudaFree(d_phiy_);
    // cudaFree(d_phiz_);
    // cudaFree(d_phi_);
    // cudaFree(d_psi_);

    // // mvc
    // cudaFree(d_Phi_);

  // somigliana
  delete[] d_PHIx_; 
  delete[] d_PHIy_; 
  delete[] d_PHIz_; 
  delete[] d_PSI_;
  delete[] d_PHI_;
  }  
  cuda_cage_precomputer(const index_t ncf,
                        const index_t ncv,
                        const index_t nv,
                        const index_t  *h_cageF,
                        const scalar_t *h_cageV,
                        const scalar_t *h_cageN,
                        const scalar_t *h_V)
                        : ncf_(ncf), ncv_(ncv), nv_(nv)
      {
    // cage and V
    d_cageF_ = new index_t[3*ncf_];
    d_cageV_ = new scalar_t[3*ncv_];
    d_cageN_ = new scalar_t[3*ncf_];
    d_V_ = new scalar_t[3*nv_];

    // copy mesh and V
    std::memcpy(d_cageF_, h_cageF, 3*ncf_*sizeof(index_t));    
    std::memcpy(d_cageV_, h_cageV, 3*ncv_*sizeof(scalar_t));
    std::memcpy(d_cageN_, h_cageN, 3*ncf_*sizeof(scalar_t));
    std::memcpy(d_V_,     h_V,     3*nv_ *sizeof(scalar_t));    

    // // basis: green
    // cudaMalloc((void**)&d_phix_, ncf*nv*sizeof(scalar_t));
    // cudaMalloc((void**)&d_phiy_, ncf*nv*sizeof(scalar_t));
    // cudaMalloc((void**)&d_phiz_, ncf*nv*sizeof(scalar_t));        
    // cudaMalloc((void**)&d_phi_,  ncv*nv*sizeof(scalar_t));
    // cudaMalloc((void**)&d_psi_,  ncf*nv*sizeof(scalar_t));    

    // // basis: mvc
    // cudaMalloc((void**)&d_Phi_, ncv*nv*sizeof(scalar_t));

    // basis: somigliana
    d_PHIx_ = new scalar_t[9*ncf_*nv_];
    d_PHIy_ = new scalar_t[9*ncf_*nv_];
    d_PHIz_ = new scalar_t[9*ncf_*nv_];

    d_PSI_ = new scalar_t[9*ncf_*nv_];
    d_PHI_ = new scalar_t[9*ncf_*nv_];
  }

  void copy_quadrature_to_device(const index_t  num,
                                 const scalar_t *h_qp,
                                 const scalar_t *h_qw) {
    nq_ = num;

    d_qp_ = new scalar_t[2 * num];
    d_qw_ = new scalar_t[num];
    
    std::memcpy(d_qp_, h_qp, 2*num*sizeof(scalar_t));
    std::memcpy(d_qw_, h_qw,   num*sizeof(scalar_t));
    }

  void precompute_somig_gpu(const scalar_t nu,
                            scalar_t *h_PHI,
                            scalar_t *h_PSI) {

    std::memset(d_PHI_,  0, 9*ncv_*nv_*sizeof(scalar_t));
    std::memset(d_PSI_,  0, 9*ncf_*nv_*sizeof(scalar_t));

    // intermediate ones
    std::memset(d_PHIx_, 0, 9*ncf_*nv_*sizeof(scalar_t));
    std::memset(d_PHIy_, 0, 9*ncf_*nv_*sizeof(scalar_t));
    std::memset(d_PHIz_, 0, 9*ncf_*nv_*sizeof(scalar_t));
    
    somig_gpu(nu, d_PHIx_, d_PHIy_, d_PHIz_, d_PSI_, d_V_, d_cageF_, d_cageV_, d_cageN_, nv_, ncf_, ncv_, d_qp_, d_qw_, nq_);
    somig_gpu_post(d_PHI_, d_PHIx_, d_PHIy_, d_PHIz_, d_cageF_, nv_, ncf_);

    std::memcpy(h_PSI, d_PSI_, 9*ncf_*nv_*sizeof(scalar_t));
    std::memcpy(h_PHI, d_PHI_, 9*ncv_*nv_*sizeof(scalar_t));
  }

 private:
  const index_t ncv_, nv_, ncf_;

  // cage on devices
  scalar_t *d_cageV_, *d_cageN_;
  index_t  *d_cageF_;

  // mesh points on devices
  scalar_t *d_V_;

  // // basis GREEN
  // scalar_t *d_phix_, *d_phiy_, *d_phiz_;
  // scalar_t *d_phi_, *d_psi_;

  // // basis MVC
  // scalar_t *d_Phi_;

  // basis SOMIGLIANA
  scalar_t *d_PHIx_, *d_PHIy_, *d_PHIz_;
  scalar_t *d_PSI_, *d_PHI_;

  // quadratures points and weights
  index_t nq_;
  scalar_t *d_qp_, *d_qw_;
};

#endif
