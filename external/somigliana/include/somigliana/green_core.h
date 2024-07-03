#ifndef GREEN_CORE_H
#define GREEN_CORE_H

#ifdef SOMIG_WITH_CUDA
#include <cuda_runtime.h>
#include "helper_cuda.h"
#endif // WITH_CUDA

#include <spdlog/spdlog.h>
#include <iostream>
#include <cstring>

typedef float scalar_t;
typedef int   index_t;

extern "C" {
  
#ifdef SOMIG_WITH_CUDA
  void mvc_gpu(scalar_t *d_PHI,
               const scalar_t *d_V,
               const index_t  *d_cageF,
               const scalar_t *d_cageV,               
               const index_t nv,
               const index_t ncf,
               const index_t ncv);
#endif // WITH_CUDA
  void mvc_cpu(scalar_t *d_PHI,
               const scalar_t *d_V,
               const index_t  *d_cageF,
               const scalar_t *d_cageV,               
               const index_t nv,
               const index_t ncf,
               const index_t ncv);

#ifdef SOMIG_WITH_CUDA
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
#endif // WITH_CUDA
  void green_cpu(scalar_t *d_phix,
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

#ifdef SOMIG_WITH_CUDA
  void green_post_gpu(scalar_t *d_phi,
                      const scalar_t *d_phix,
                      const scalar_t *d_phiy,
                      const scalar_t *d_phiz,
                      const index_t  *d_cageF,
                      const index_t nv,
                      const index_t ncf,
                      const index_t ncv);  
#endif // WITH_CUDA
  void green_post_cpu(scalar_t *d_phi,
                      const scalar_t *d_phix,
                      const scalar_t *d_phiy,
                      const scalar_t *d_phiz,
                      const index_t  *d_cageF,
                      const index_t nv,
                      const index_t ncf,
                      const index_t ncv);  

#ifdef SOMIG_WITH_CUDA
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
#endif // WITH_CUDA
  void somig_cpu(const scalar_t nu,
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

#ifdef SOMIG_WITH_CUDA
  // reduce PHIxyz to PHI
  void somig_post_gpu(scalar_t *d_PHI,
                      const scalar_t *d_PHIx,
                      const scalar_t *d_PHIy,
                      const scalar_t *d_PHIz,
                      const index_t  *d_cageF,
                      const index_t nv,
                      const index_t ncf);
#endif // WITH_CUDA
  void somig_post_cpu(scalar_t *d_PHI,
                      const scalar_t *d_PHIx,
                      const scalar_t *d_PHIy,
                      const scalar_t *d_PHIz,
                      const index_t  *d_cageF,
                      const index_t nv,
                      const index_t ncf);
}

void mvc_compute(scalar_t *d_PHI,
               const scalar_t *d_V,
               const index_t  *d_cageF,
               const scalar_t *d_cageV,               
               const index_t nv,
               const index_t ncf,
               const index_t ncv){
#ifdef SOMIG_WITH_CUDA
  mvc_gpu(d_PHI,
              d_V,
              d_cageF,
              d_cageV,               
              nv,
              ncf,
              ncv);
#else
  mvc_cpu(d_PHI,
              d_V,
              d_cageF,
              d_cageV,               
              nv,
              ncf,
              ncv);
#endif
  }

void green_compute(scalar_t *d_phix,
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
                const index_t nq){
#ifdef SOMIG_WITH_CUDA
  green_gpu(d_phix,
            d_phiy,
            d_phiz,
            d_psi,
            d_V,
            d_cageF,
            d_cageV,
            d_cageN,
            nv,
            ncf,
            ncv,
            d_qp,
            d_qw,
            nq);

#else
  green_cpu(d_phix,
            d_phiy,
            d_phiz,
            d_psi,
            d_V,
            d_cageF,
            d_cageV,
            d_cageN,
            nv,
            ncf,
            ncv,
            d_qp,
            d_qw,
            nq);
#endif
}

void green_post(scalar_t *d_phi,
                    const scalar_t *d_phix,
                    const scalar_t *d_phiy,
                    const scalar_t *d_phiz,
                    const index_t  *d_cageF,
                    const index_t nv,
                    const index_t ncf,
                    const index_t ncv){
#ifdef SOMIG_WITH_CUDA
  green_post_gpu(d_phi,
                  d_phix,
                  d_phiy,
                  d_phiz,
                  d_cageF,
                  nv,
                  ncf,
                  ncv);  
#else
  green_post_cpu(d_phi,
                  d_phix,
                  d_phiy,
                  d_phiz,
                  d_cageF,
                  nv,
                  ncf,
                  ncv); 
#endif                                            
}
  
void somig_compute(const scalar_t nu,
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
                const index_t nq){
#ifdef SOMIG_WITH_CUDA
  somig_gpu(nu,
            d_PHIx,
            d_PHIy,
            d_PHIz,
            d_PSI ,
            d_V,
            d_cageF,
            d_cageV,
            d_cageN,
            nv,
            ncf,
            ncv,
            d_qp,
            d_qw,
            nq);   
#else
  somig_cpu(nu,
            d_PHIx,
            d_PHIy,
            d_PHIz,
            d_PSI ,
            d_V,
            d_cageF,
            d_cageV,
            d_cageN,
            nv,
            ncf,
            ncv,
            d_qp,
            d_qw,
            nq);                  
#endif // WITH_CUDA
}

void somig_post(scalar_t *d_PHI,
                const scalar_t *d_PHIx,
                const scalar_t *d_PHIy,
                const scalar_t *d_PHIz,
                const index_t  *d_cageF,
                const index_t nv,
                const index_t ncf){
#ifdef SOMIG_WITH_CUDA
  somig_post_gpu(d_PHI,
                    d_PHIx,
                    d_PHIy,
                    d_PHIz,
                    d_cageF,
                    nv,
                    ncf);
#else // WITH_CUDA
  somig_post_cpu(d_PHI,
                    d_PHIx,
                    d_PHIy,
                    d_PHIz,
                    d_cageF,
                    nv,
                    ncf);
#endif    
  }

// TODO: Fallback to CPU if no CUDA device is found
class cage_precomputer
{
 public:
 ~cage_precomputer() {
#ifdef SOMIG_WITH_CUDA
    // mesh and points
    cudaFree(d_cageF_);
    cudaFree(d_cageV_);
    cudaFree(d_cageN_);
    cudaFree(d_V_);

    // quadratures
    cudaFree(d_qp_);
    cudaFree(d_qw_);

    // // green
    // cudaFree(d_phix_);
    // cudaFree(d_phiy_);
    // cudaFree(d_phiz_);
    // cudaFree(d_phi_);
    // cudaFree(d_psi_);

    // // mvc
    // cudaFree(d_Phi_);

    // somigliana
    cudaFree(d_PHIx_); 
    cudaFree(d_PHIy_); 
    cudaFree(d_PHIz_); 
    cudaFree(d_PSI_);
    cudaFree(d_PHI_);
#else

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
#endif // WITH_CUDA
  }  
  
  cage_precomputer(const index_t ncf,
                        const index_t ncv,
                        const index_t nv,
                        const index_t  *h_cageF,
                        const scalar_t *h_cageV,
                        const scalar_t *h_cageN,
                        const scalar_t *h_V)
                        : ncf_(ncf), ncv_(ncv), nv_(nv)
      {
#ifdef SOMIG_WITH_CUDA
    index_t devID;
    devID = findCudaDevice(0, NULL);

    if (devID < 0) {
      spdlog::error("No CUDA Capable devices found");
      exit(EXIT_SUCCESS);
    } else {
      spdlog::info("device ID={}", devID);
    }

    // cage and V
    cudaMalloc((void**)&d_cageF_, 3*ncf*sizeof(index_t));
    cudaMalloc((void**)&d_cageV_, 3*ncv*sizeof(scalar_t));
    cudaMalloc((void**)&d_cageN_, 3*ncf*sizeof(scalar_t));
    cudaMalloc((void**)&d_V_,     3*nv *sizeof(scalar_t));

    // copy mesh and V
    cudaMemcpy(d_cageF_, h_cageF, 3*ncf_*sizeof(index_t),  cudaMemcpyHostToDevice);    
    cudaMemcpy(d_cageV_, h_cageV, 3*ncv_*sizeof(scalar_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cageN_, h_cageN, 3*ncf_*sizeof(scalar_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_V_,     h_V,     3*nv_ *sizeof(scalar_t), cudaMemcpyHostToDevice);    

    // // basis: green
    // cudaMalloc((void**)&d_phix_, ncf*nv*sizeof(scalar_t));
    // cudaMalloc((void**)&d_phiy_, ncf*nv*sizeof(scalar_t));
    // cudaMalloc((void**)&d_phiz_, ncf*nv*sizeof(scalar_t));        
    // cudaMalloc((void**)&d_phi_,  ncv*nv*sizeof(scalar_t));
    // cudaMalloc((void**)&d_psi_,  ncf*nv*sizeof(scalar_t));    

    // // basis: mvc
    // cudaMalloc((void**)&d_Phi_, ncv*nv*sizeof(scalar_t));

    // basis: somigliana
    cudaMalloc((void**)&d_PHIx_, 9*ncf*nv*sizeof(scalar_t)); 
    cudaMalloc((void**)&d_PHIy_, 9*ncf*nv*sizeof(scalar_t)); 
    cudaMalloc((void**)&d_PHIz_, 9*ncf*nv*sizeof(scalar_t)); 
    cudaMalloc((void**)&d_PSI_,  9*ncf*nv*sizeof(scalar_t));
    cudaMalloc((void**)&d_PHI_,  9*ncv*nv*sizeof(scalar_t));
#else
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
#endif // SOMIG_WITH_CUDA
  }

  void copy_quadrature_to_device(const index_t  num,
                                 const scalar_t *h_qp,
                                 const scalar_t *h_qw) {
    nq_ = num;
#ifdef SOMIG_WITH_CUDA

    cudaMalloc((void**)&d_qp_, 2*num*sizeof(scalar_t));
    cudaMalloc((void**)&d_qw_,   num*sizeof(scalar_t));
    
    cudaMemcpy(d_qp_, h_qp, 2*num*sizeof(scalar_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_qw_, h_qw,   num*sizeof(scalar_t), cudaMemcpyHostToDevice);
#else
    d_qp_ = new scalar_t[2 * num];
    d_qw_ = new scalar_t[num];
    
    std::memcpy(d_qp_, h_qp, 2*num*sizeof(scalar_t));
    std::memcpy(d_qw_, h_qw,   num*sizeof(scalar_t));
#endif // SOMIG_WITH_CUDA
    }

  void precompute_somig(const scalar_t nu,
                            scalar_t *h_PHI,
                            scalar_t *h_PSI) {
#ifdef SOMIG_WITH_CUDA
    cudaMemset(d_PHI_,  0, 9*ncv_*nv_*sizeof(scalar_t));
    cudaMemset(d_PSI_,  0, 9*ncf_*nv_*sizeof(scalar_t));

    // intermediate ones
    cudaMemset(d_PHIx_, 0, 9*ncf_*nv_*sizeof(scalar_t));
    cudaMemset(d_PHIy_, 0, 9*ncf_*nv_*sizeof(scalar_t));
    cudaMemset(d_PHIz_, 0, 9*ncf_*nv_*sizeof(scalar_t));
    somig_compute(nu, d_PHIx_, d_PHIy_, d_PHIz_, d_PSI_, d_V_, d_cageF_, d_cageV_, d_cageN_, nv_, ncf_, ncv_, d_qp_, d_qw_, nq_);
    somig_post(d_PHI_, d_PHIx_, d_PHIy_, d_PHIz_, d_cageF_, nv_, ncf_);

    cudaMemcpy(h_PSI, d_PSI_, 9*ncf_*nv_*sizeof(scalar_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_PHI, d_PHI_, 9*ncv_*nv_*sizeof(scalar_t), cudaMemcpyDeviceToHost);
#else
    std::memset(d_PHI_,  0, 9*ncv_*nv_*sizeof(scalar_t));
    std::memset(d_PSI_,  0, 9*ncf_*nv_*sizeof(scalar_t));

    // intermediate ones
    std::memset(d_PHIx_, 0, 9*ncf_*nv_*sizeof(scalar_t));
    std::memset(d_PHIy_, 0, 9*ncf_*nv_*sizeof(scalar_t));
    std::memset(d_PHIz_, 0, 9*ncf_*nv_*sizeof(scalar_t));
    
    somig_compute(nu, d_PHIx_, d_PHIy_, d_PHIz_, d_PSI_, d_V_, d_cageF_, d_cageV_, d_cageN_, nv_, ncf_, ncv_, d_qp_, d_qw_, nq_);
    somig_post(d_PHI_, d_PHIx_, d_PHIy_, d_PHIz_, d_cageF_, nv_, ncf_);

    std::memcpy(h_PSI, d_PSI_, 9*ncf_*nv_*sizeof(scalar_t));
    std::memcpy(h_PHI, d_PHI_, 9*ncv_*nv_*sizeof(scalar_t));
  #endif // SOMIG_WITH_CUDA

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
