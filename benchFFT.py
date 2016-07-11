#!/usr/bin/env python

# References:
#
# http://software.intel.com/en-us/intel-mkl

import time

import numpy
import numpy.fft
import numpy.fft.fftpack

#from scipy import fftpack
import scipy
import scipy.fftpack

#from scipy.fftpack import fft
import os

from ctypes import *
import ctypes as _ctypes

import argparse
import sys

#import matplotlib as mpl
#mpl.use('Agg')
#import matplotlib.pyplot as plt
#from matplotlib import ticker


def show_info():
    try:
        import mkl
        print("MKL MAX THREADS:", mkl.get_max_threads())
    except ImportError:
        print("MKL NOT INSTALLED")


def plot_results(datas, factor=None, algo='FFT'):
    xlabel = r'Array Size (2^x)'
    ylabel = 'Speed (GFLOPs)'
    backends = ['numpy', 'numpy+mkl']

    plt.clf()
    fig1, ax1 = plt.subplots()
    plt.figtext(0.90, 0.94, "Note: higher is better", va='top', ha='right')
    w, h = fig1.get_size_inches()
    fig1.set_size_inches(w*1.5, h)
    ax1.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
    ax1.get_xaxis().set_minor_locator(ticker.NullLocator())
    ax1.set_xticks(datas[0][:,0])
    ax1.grid(color="lightgrey", linestyle="--", linewidth=1, alpha=0.5)
    if factor:
        ax1.set_xticklabels([str(int(x)) for x in datas[0][:,0]/factor])
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.xlim(datas[0][0,0]*.9, datas[0][-1,0]*1.025)
    plt.suptitle("%s Performance" % ("FFT"), fontsize=28)

    for backend, data in zip(backends, datas):
        N = data[:, 0]
        plt.plot(N, data[:, 1], 'o-', linewidth=2, markersize=5, label=backend)
        plt.legend(loc='upper left', fontsize=18)

    plt.savefig(algo + '.png')
    
    
def run_scipy(repeat, size, myprecision, mkl=True):
    args = tuple(1 * [size])
    #a = numpy.random.randn(*args) + 1j * numpy.random.randn(*args)


    #if myprecision == 'single':
    #    a = a.astype(numpy.complex64)
    #elif myprecision == 'double':
    #    a = a.astype(numpy.complex128)

    if myprecision == 'single':
        a = numpy.array(numpy.random.random(*args), dtype=numpy.complex64, order='C', copy=False)
    else:
        a = numpy.array(numpy.random.random(*args), dtype=numpy.complex128, order='C', copy=False)

    #warmup MKL
    #calculate warm up time (setup_time)
    #warmup_init = time.time()
    #x = numpy.fft.fftn(a)
    #x = numpy.fft.fftn(a)
    #warmup_total = time.time() - warmup_init
    b = scipy.fftpack.fft(a)

    start_time = time.time()
    for dummy in xrange(repeat):
        if mkl: 
            #b = numpy.fft.fft(a)
            b = scipy.fftpack.fft(a)
    time_taken = time.time() - start_time


    return time_taken

def run(repeat, size, myprecision, mkl=True):
    args = tuple(1 * [size])
    #a = numpy.random.randn(*args) + 1j * numpy.random.randn(*args)


    #if myprecision == 'single':
    #    a = a.astype(numpy.complex64)
    #elif myprecision == 'double':
    #    a = a.astype(numpy.complex128)

    if myprecision == 'single':
        a = numpy.array(numpy.random.random(*args), dtype=numpy.complex64, order='C', copy=False)
    else:
        a = numpy.array(numpy.random.random(*args), dtype=numpy.complex128, order='C', copy=False)

    #warmup MKL
    #calculate warm up time (setup_time)
    #warmup_init = time.time()
    #x = numpy.fft.fftn(a)
    #x = numpy.fft.fftn(a)
    #warmup_total = time.time() - warmup_init

    b = numpy.fft.fft(a)

    start_time = time.time()
    for dummy in xrange(repeat):
        if mkl: 
            b = numpy.fft.fft(a)
        else:
            b = numpy.fft.fftpack.fft(a)
    time_taken = time.time() - start_time


    return time_taken


def run2d(repeat, size, myprecision, mkl=True):
    args = size
    #a = numpy.random.randn(*args) + 1j * numpy.random.randn(*args)


    if myprecision == 'single':
        a = numpy.array(numpy.random.random(size), dtype=numpy.complex64, order='C', copy=False)
    else:
        a = numpy.array(numpy.random.random(size), dtype=numpy.complex128, order='C', copy=False)
    """
    if myprecision == 'single':
        a = a.astype(numpy.complex64)
    elif myprecision == 'double':
        a = a.astype(numpy.complex128)
    """

    #warmup MKL
    #x = numpy.fft.fftn(a)

    start_time = time.time()
    for dummy in xrange(repeat):
        if mkl:
            #from numpy import fft_intel
            #b = numpy.fft_intel.fft2(a)
            b = numpy.fft.fftn(a)
        else:
            b = numpy.fft.fftpack.fftn(a)
    time_taken = time.time() - start_time
    return time_taken


def oneDFFT(myprecision, mythreads, ispsf=False):


    dataMKL = []
    dataNoMKL = []

    #print("\nMKL_NUM_THREADS = %s" %  os.environ['MKL_NUM_THREADS'])

    print("\n%s Precision complex to complex 1D" % myprecision)


    #print('\n%8s , %8s , %8s , %12s , %16s , %16s , %16s ,  %16s  , %16s , %16s, %16s' % ('trials', '2^n', 'Threads', 'array size', 'time(micro s) MKL', 'time(micro s) No MKL', 'Setup_time micro s', 'GFLOPs (MKL)', 'GFLOPS + Setup time' , 'GFLOPs (No MKL)', 'GFLOPS + setuptime (No MKL)'))
    print('\n%8s , %8s , %8s , %12s , %16s,  %16s , %16s , %16s ,  %16s ' % ('trials', '2^n', 'Threads', 'array size', 'Total Time all trials' , 'time(micro s) MKL', 'time(micro s) No MKL', 'GFLOPs (MKL)', 'GFLOPs (No MKL)'))
    print( '------------------------------------------'*2)


    print("Results for NumPy")
    for n in xrange(4, 25):
        size = 2 ** n
        # to keep the experiment from taking too long
        if n < 10:
            trials = 1000
        elif n < 20:
            trials = 100
        else:
            trials = 15
        

        #n = 19
        #size = 2 ** 19
        #trials = 10
        #mflop = 5.0*size*numpy.log10(size)    
        mflop = 5.0*size*numpy.log2(size)    
        gflop = mflop / 1000.
        
        s = run(trials, size, myprecision)
        #time in microseconds
        avg_ms = (s/float(trials)) * 1000000.
        
        dataMKL.append(numpy.asarray([n, gflop/avg_ms ]))
        #time in flop calculation is for the average time. The setup time for MKL
        #it is not included in the calculations.
        gflops_mkl = gflop/avg_ms
        #flops calculation considering the setup time. Setup time should not be included
        #in calculations because it not affecting the fft transformation time.
        #gflops_mkl_setuptime = gflop/(avg_ms + setup_time*1000000)
        
        if ispsf:
            s2 = 1.
        else:
            s2 = run(trials, size, myprecision, mkl=False)
        
        avg_ms = (s2/float(trials)) * 1000000.
        gflops_nomkl = gflop/avg_ms
        #gflops_nomkl_setuptime = gflop/(avg_ms+setup_time_nomkl*1000000)
        dataNoMKL.append(numpy.asarray([n, gflop/avg_ms ]))
        print('%8i , %8i  , %8s , %12i , %16.4f , %16.4e , %16.4e  , %16.4e , %16.4e' % (trials, n, mythreads, size, s*1000000. , (s/trials)*1000000., (s2/trials)*1000000., gflops_mkl, gflops_nomkl))


    print("Results for Scipy")


    for n in xrange(4, 25):
        size = 2 ** n
        # to keep the experiment from taking too long
        if n < 10:
            trials = 1000
        elif n < 20:
            trials = 100
        else:
            trials = 15
            
        NAN = 99.
        #mflop = 5.0*size*numpy.log10(size)    
        mflop = 5.0*size*numpy.log2(size)    
        gflop = mflop / 1000.

        s_scipy = run_scipy(trials, size, myprecision)
        #time in microseconds
        avg_ms = (s_scipy/float(trials)) * 1000000.
        gflops_mkl = gflop/avg_ms
        print('%8i , %8i  , %8s , %12i , %16.4f , %16.4e , %16.4e  , %16.4e , %16.4e' % (trials, n, mythreads, size, s_scipy*1000000., (s_scipy/trials)*1000000., NAN, gflops_mkl, gflops_nomkl))

def twoDFFT(myprecision, mythreads):

    dataMKL = []
    dataNoMKL = []

    #2D aarays from MKL team:
    matrices = [(32,32), (64,64), (512,16), (128,64), (64,128), (512,64), (16,512), (128,128), (256,128), (256,256), (64,512), (512,512), (64,1024), (256,1024), (1024,1024), (2048,256), (1024,128), (2048,2048), (512,4096), (8192,64), (16384,16), (16384,32), (16384,64), (16384,16384)]

    newmatrices = []
    #generate 2D matrices with 2^n from n=1-24
    for n in xrange(4,15):
        row = 2**n
        for m in xrange(4,15):
            col = 2**m
            if n < m:
                continue
            newmatrices.append((row,col))

    print("\nMKL_NUM_THREADS = %s" %  os.environ['MKL_NUM_THREADS'])
        
    print("\n%s Precision complex to complex 2D" % myprecision)

    print('\n%8s , %8s , %12s , %16s , %16s , %16s , %16s' % ('trials', 'Threads', 'array size', 'time(s) MKL', 'time(s) No MKL', 'GFLOPs (MKL)', 'GFLOPs (No MKL)'))
    print ( '------------------------------------------'*2)

    #do 2D arrays:
    n=5
    for value in newmatrices:
        size = value
        #trials = 8  #this is done by MKL team
        
        if n < 8 and m < 8:
            trials = 10
        elif n < 15 and m < 15:
            trials = 6
        else:
            trials = 3

        trials = 3
        totalsize = size[0]*size[1]

        mflop = 5.0*totalsize*numpy.log2(totalsize)    
        gglop = mflop / 1000

        
        s = run2d(trials, size, myprecision)
        avg_ms = (s/trials) * 1000000
        dataMKL.append(numpy.asarray([size, gglop/avg_ms ]))    
        gflops_mkl = gglop/avg_ms
        

        s2 = run2d(trials, size, myprecision, mkl=False)
        avg_ms = (s2/trials) * 1000000
        gflops_nomkl = gglop/avg_ms
        dataNoMKL.append(numpy.asarray([size, gglop/avg_ms ]))
        print('%8i , %8s , %6ix%i , %12.4f , %12.4f , %12.4f , %12.4f' % (trials, mythreads, size[0],size[1], s, s2, gflops_mkl, gflops_nomkl))
        #print '%8i , %8s , %6ix%i , %12.4f , %12.4f '  % (trials, mythreads, size[0],size[1],  s2, gflops_nomkl)


def threeDFFT(myprecision, mythreads):

    dataMKL = []
    dataNoMKL = []

    #2D aarays from MKL team:
    matrices = [(32,32,32), (64,64,64), (256,64,32), (128,128,64), (128,128,128), (128,64,256), (512,128,64), (256,128,256), (256,256,256), (512,64,1024), (512,256,512), (512,512,512)]

    print("\nMKL_NUM_THREADS = %s" %  os.environ['MKL_NUM_THREADS'])
        
    print("\n%s Precision complex to complex 2D" % myprecision)

    print('\n%8s , %8s , %12s , %16s , %16s , %16s , %16s' % ('trials', 'Threads', 'array size', 'time(s) MKL', 'time(s) No MKL', 'GFLOPs (MKL)', 'GFLOPs (No MKL)'))
    print ( '------------------------------------------'*2)

    #do 2D arrays:
    n=1
    for value in matrices:
        size = value
        trials = 8  #this is done by MKL team

        totalsize = size[0]*size[1]*size[2]

        mflop = 5.0*totalsize*numpy.log2(totalsize)    
        gglop = mflop / 1000

        s = run2d(trials, size, myprecision)
        avg_ms = (s/trials) * 1000000
        dataMKL.append(numpy.asarray([n, gglop/avg_ms ]))    
        gflops_mkl = gglop/avg_ms

        s2 = run2d(trials, size, myprecision, mkl=False)
        avg_ms = (s2/trials) * 1000000
        gflops_nomkl = gglop/avg_ms
        dataNoMKL.append(numpy.asarray([n, gglop/avg_ms ]))
        print('%8i , %8s , %6ix%ix%i , %12.4f , %12.4f , %12.4f , %12.4f' % (trials, mythreads, size[0],size[1], size[2],s, s2, gflops_mkl, gflops_nomkl))


def run_mkl_direct(size, trials, locmkl):

    args = tuple(1*[size])
    a = numpy.array(numpy.random.random(*args), dtype=numpy.complex64, order='C', copy=False)
    desc_handle = _ctypes.c_void_p(0)
    dims = (_ctypes.c_long*1)(*a.shape)
    _DFTI_SINGLE = _ctypes.c_int(35)
    _DFTI_COMPLEX = _ctypes.c_int(32)

    locmkl.DftiCreateDescriptor(_ctypes.byref(desc_handle), _DFTI_SINGLE, _DFTI_COMPLEX, _ctypes.c_long(1), dims )
    locmkl.DftiCommitDescriptor(desc_handle)
    
    #warm up
    locmkl.DftiComputeForward(desc_handle, a.ctypes.data_as(_ctypes.c_void_p))
    locmkl.DftiComputeForward(desc_handle, a.ctypes.data_as(_ctypes.c_void_p))

    tic = time.time() 
    for dummy in xrange(trials):
        locmkl.DftiComputeForward(desc_handle, a.ctypes.data_as(_ctypes.c_void_p))
    toc = time.time() - tic

    return toc

def call_direct_mkl(distro, threads):

    #a = np.ones(520000, dtype=np.complex64)
    
    print('Trials ,   N , Matrix Size (=2^N) , Total Time all Trials [micro s], Ave Time [micro s], GFLOPs   ')
    
    if distro == 'anaconda':
        print("Loading MKL Libraries fro Anaconda")
        print('/nfs/fx/proj/scripting_tools/racovarr/Continuum/anaconda27_mkl/pkgs/mkl-rt-11.1-p0/lib/libmkl_rt.so')
        themkl = _ctypes.cdll.LoadLibrary('/nfs/fx/proj/scripting_tools/racovarr/Continuum/anaconda27_mkl/pkgs/mkl-rt-11.1-p0/lib/libmkl_rt.so')
    
    else:
        #mkl = _ctypes.cdll.LoadLibrary('./build/ext/lib/libmkl_rt.so')
        print("Loadiing MKL libraries from Intel distro")
        print('/nfs/fx/disks/fx_home_disk1/racovarr/intel/intelpython27/ext/lib/libmkl_rt.so')
        themkl = _ctypes.cdll.LoadLibrary('/nfs/fx/disks/fx_home_disk1/racovarr/intel/intelpython27/ext/lib/libmkl_rt.so')


    #themkl.MKL_Set_Num_Threads(_ctypes.c_int(threads))
    #themkl.MKL_Set_Num_Threads(threads)
    #print("num theads sets", themkl.mkl_get_max_threads(32))

    #for n in xrange(4, 25):
    for n in xrange(18, 20):
        size = 2 ** n
        # to keep the experiment from taking too long
        #if n < 10:
        #    trials = 1000
        #elif n < 20:
        #    trials = 100
        #else:
        trials = 15

        #n = 19
        #trials = 10
        #size = 2 ** 19
        #mflop = 5.0*size*numpy.log10(size)    
        mflop = 5.0*size*numpy.log2(size)    
        gflop = mflop / 1000.
    
        s = run_mkl_direct(size, trials, themkl)

        print("time for all trials = %14.2f [micro sec], for %i trials  " % (s*1000000., trials))
        

        avg_ms = (s/float(trials)) * 1000000.
        gflops_mkl = gflop/avg_ms
        
        #if n == 19:

        #else:
        print('%8i , %8i , %8i , %12.4f, %12.4f , %12.4f ' % (trials, n, size, s*1000000., avg_ms , gflops_mkl))

    

def main():  

    parser = argparse.ArgumentParser("Benchmark for NumPy Matrix Multiplication, Cholesky, LU and Single Value decomposition.")
    parser.add_argument("-a", "--anaconda", help="Use when running Anaconda version", action="store_true")
    parser.add_argument("-i", "--intel", help="Use when running Intel version", action="store_true")
    parser.add_argument("-p", "--psf", help="Use when running PSF version", action="store_true")
    args = parser.parse_args()



    #Number of Threads
    #threads = ['1', '4', '16', '32']
    #precission = ['single','double']
    precission = ['single']

         
     
    """
    for num in threads:
        if num == '1':
            os.environ['MKL_NUM_THREADS'] = num
        elif num == '4':
            os.environ['MKL_NUM_THREADS'] = num
        elif num == '16':
            os.environ['MKL_NUM_THREADS'] = num
        else:
            os.environ['MKL_NUM_THREADS'] = num
             
        print("mkl ....",os.environ['MKL_NUM_THREADS'])
    """
    try:
        threads = os.environ['MKL_NUM_THREADS']
    except KeyError as e:
        print('KeyError - reason "%s"' % str(e))
    try:
        dynamics = os.environ['MKL_DYNAMIC']
    except KeyError as e:
        print('KeyError - reason "%s"' % str(e))
    try:
        affinity = os.environ['KMP_AFFINITY']
    except KeyError as e:
        print('KeyError - reason "%s"' % str(e))

    myPythonVersion = sys.version
    mynumpy = numpy.__version__
    myscipy = scipy.__version__

    print("Numpy version =", mynumpy)
    print("SciPy version =", myscipy)

    
    #threadingLayer = os.environ['MKL_THREADING_LAYER']
    #allDomain = os.environ['MKL_DOMAIN_NUM_THREADS']

    
    #print("\n Enironment: MKL_NUM_THREADS=%s , MKL_DYNAMIC=%s , KMP_AFFINITY=%s , MKL_THREADING_LAYER=%s \n MKL_DOMAIN+NUM+THREADS=%s" % (threads, dynamics, affinity, threadingLayer, allDomain))
    print("\n Enironment: MKL_NUM_THREADS=%s , MKL_DYNAMIC=%s , KMP_AFFINITY=%s \n Python Version=%s" % (threads, dynamics, affinity, sys.version))
    print("\n Python version %s \n" % myPythonVersion)
  
    if args.anaconda:
        #for Anaconda
        anamkl = cdll.LoadLibrary('/nfs/fx/proj/scripting_tools/racovarr/Continuum/anaconda27_mkl/pkgs/mkl-11.3.1-0/lib/libmkl_rt.so')
        vs = create_string_buffer(200)
        anamkl.MKL_Get_Version_String(vs, c_int(200))
        print(vs.value)
        
        call_direct_mkl('anaconda', threads)
        import mkl

    if args.intel:
        #for intel
        intmkl = cdll.LoadLibrary('/nfs/fx/disks/fx_home_disk1/racovarr/intel/intelpython27/ext/lib/libmkl_rt.so')
        vs = create_string_buffer(200)
        intmkl.MKL_Get_Version_String(vs, c_int(200))
        print(vs.value)

        call_direct_mkl('intel',threads)


    #Do single or double precission calculations
    for prec in precission:
        if args.psf:
            oneDFFT(prec, os.environ['MKL_NUM_THREADS'], ispsf=True)
        else:
            oneDFFT(prec, os.environ['MKL_NUM_THREADS'])
        

        #twoDFFT(prec,os.environ['MKL_NUM_THREADS'])
        # 3D has not been tested
        #threeDFFT(prec,os.environ['MKL_NUM_THREADS'])

    #print(vs.value)

    #     datas = numpy.asarray([numpy.asarray(dataNoMKL),numpy.asarray(dataMKL)])
     
    #     plot_results(datas,algo='FFT')
     


if __name__ == '__main__':
    main()
