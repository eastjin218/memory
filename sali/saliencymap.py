import cv2
import tensorflow as tf
import math, os, glob
import numpy as np
import numpy.matlib
import scipy.io
import sklearn.preprocessing

class SaliencyMap():
    def __init__(self, input_img):
        self.input_img = input_img
    
    def setupParams(self):
        gaborparams = {
            'stddev': 2,
            'elongation': 2,
            'filterSize': -1,
            'filterPeriod': np.pi
        }

        params = {
            'gaborparams': gaborparams,
            'sigma_frac_act': 0.15,
            'sigma_frac_norm': 0.06,
            'max_level': 4,
            'thetas': [0, 45, 90, 135]
        }

        return params

    def compute_saliency(self):
        input_img = cv2.imread(self.input_img)
        
        params = self.setupParams()
        b= np.asarray(input_img[:,:,0], dtype='float32')
        g= np.asarray(input_img[:,:,1], dtype='float32')
        r= np.asarray(input_img[:,:,2], dtype='float32')
        L= np.asarray(tf.math.maximum(tf.math.maximum(r,g),b), dtype='float32')
        b_pyr = self.getPyramids(b, params['max_level'])
        g_pyr = self.getPyramids(g, params['max_level'])
        r_pyr = self.getPyramids(r, params['max_level'])
        L_pyr = self.getPyramids(L, params['max_level'])
        
        featMaps = {
            0:[],
            1:[],
            2:[],
            3:[]
        }
        for i in range(0, len(b_pyr)):
            p_r = r_pyr[i]
            p_g = g_pyr[i]
            p_b = b_pyr[i]
            p_L = L_pyr[i]

            maps = self.calculateFeatureMaps(p_r, p_g, p_b, p_L, params)

            for i in range(0,3):
                resized_m = cv2.resize(maps[i], (32, 28), interpolation=cv2.INTER_CUBIC)
                featMaps[i].append(resized_m)
            for m in maps[3]:
                resized_m = cv2.resize(m, (32, 28), interpolation=cv2.INTER_CUBIC)
                featMaps[3].append(resized_m)
        activationMaps = []
        activation_sigma = params['sigma_frac_act']*np.mean([32, 28]) # the shape of map

        for i in range(0,4):
            for map in featMaps[i]:
                activationMaps.append(self.calculate(map, activation_sigma))
        normalisedActivationMaps = []
        normalisation_sigma = params['sigma_frac_norm']*np.mean([32, 28])

        for map in activationMaps:
            normalisedActivationMaps.append(self.normalize(map, normalisation_sigma))


        mastermap = normalisedActivationMaps[0]
        for i in range(1, len(normalisedActivationMaps)):
            mastermap = np.add(normalisedActivationMaps[i], mastermap)

        gray = cv2.normalize(mastermap, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        mastermap_res = cv2.resize(gray, (input_img.shape[1], input_img.shape[0]), interpolation=cv2.INTER_CUBIC)

        return mastermap_res*255.0
    
    def loadGraphDistanceMatrixFor28x32(self):
        f = scipy.io.loadmat("./28__32__m__2.mat")
        distanceMat = np.array(f['grframe'])[0][0][0]
        lx = np.array(f['grframe'])[0][0][1]
        dim = np.array(f['grframe'])[0][0][2]
        return [distanceMat, lx, dim]

    def calculate(self, map, sigma):
        [distanceMat, _, _] = self.loadGraphDistanceMatrixFor28x32()
        denom = 2 * pow(sigma, 2)
        expr = -np.divide(distanceMat, denom)
        Fab = np.exp(expr)
        
        map_linear = np.ravel(map, order='F')  # column major
    
        state_transition_matrix = Fab * np.abs(
            (np.zeros((distanceMat.shape[0], distanceMat.shape[1])) + map_linear).T - map_linear
        ).T

        norm_STM = sklearn.preprocessing.normalize(state_transition_matrix, axis=0, norm='l1')

        eVec = self.markovchain(norm_STM, 0.0001)
        processed_reshaped = np.reshape(eVec, map.shape, order='F')

        return processed_reshaped

    def normalize(self, map, sigma):
        [distanceMat, _, _] = self.loadGraphDistanceMatrixFor28x32()
        denom = 2 * pow(sigma, 2)
        expr = -np.divide(distanceMat, denom)
        Fab = np.exp(expr)

        map_linear = np.ravel(map, order='F') 
        state_transition_matrix = (Fab.T * np.abs(map_linear)).T

        norm_STM = sklearn.preprocessing.normalize(state_transition_matrix, axis=0, norm='l1')

        eVec = self.markovchain(norm_STM, 0.0001)
        processed_reshaped = np.reshape(eVec, map.shape, order='F')

        return processed_reshaped
    
    def markovchain(self, mat, tolerance):
        w,h = mat.shape
        diff = 1
        v = np.divide(np.ones((w, 1), dtype=np.float32), w)
        oldv = v
        oldoldv = v

        while diff > tolerance :
            oldv = v
            oldoldv = oldv
            v = np.dot(mat,v)
            diff = np.linalg.norm(oldv - v, ord=2)
            s = sum(v)
            if s>=0 and s< np.inf:
                continue
            else:
                v = oldoldv
                break

        v = np.divide(v, sum(v))

        return v
                
    def getPyramids(self, image, max_level):
        imagePyr = [cv2.pyrDown(image)]
        for i in range(1, max_level):
            imagePyr.append(cv2.pyrDown(imagePyr[i-1]))
        return imagePyr[1:]
    
    def getGaborKernels(self, gaborparams, thetas):
        gaborKernels = {}
        def getGaborKernel(gaborparams, angle, phase):
            gp = gaborparams
            major_sd = gp['stddev']
            minor_sd = major_sd * gp['elongation']
            max_sd = max(major_sd, minor_sd)

            sz = gp['filterSize']
            if sz == -1:
                sz = math.ceil(max_sd * math.sqrt(10))
            else:
                sz = math.floor(sz / 2)

            psi = np.pi / 180 * phase
            rtDeg = np.pi / 180 * angle

            omega = 2 * np.pi / gp['filterPeriod']
            co = math.cos(rtDeg)
            si = -math.sin(rtDeg)
            major_sigq = 2 * pow(major_sd, 2)
            minor_sigq = 2 * pow(minor_sd, 2)

            vec = range(-int(sz), int(sz) + 1)
            vlen = len(vec)
            vco = [i * co for i in vec]
            vsi = [i * si for i in vec]

            # major = np.matlib.repmat(np.asarray(vco).transpose(), 1, vlen) + np.matlib.repmat(vsi, vlen, 1)
            a = np.tile(np.asarray(vco).transpose(), (vlen, 1)).transpose()
            b = np.matlib.repmat(vsi, vlen, 1)
            major = a + b
            major2 = np.power(major, 2)

            # minor = np.matlib.repmat(np.asarray(vsi).transpose(), 1, vlen) - np.matlib.repmat(vco, vlen, 1)
            a = np.tile(np.asarray(vsi).transpose(), (vlen, 1)).transpose()
            b = np.matlib.repmat(vco, vlen, 1)
            minor = a + b
            minor2 = np.power(minor, 2)

            a = np.cos(omega * major + psi)
            b = np.exp(-major2 / major_sigq - minor2 / minor_sigq)
            # result = np.cos(omega * major + psi) * exp(-major2/major_sigq - minor2/minor_sigq)
            result = np.multiply(a, b)

            filter1 = np.subtract(result, np.mean(result.reshape(-1)))
            filter1 = np.divide(filter1, np.sqrt(np.sum(np.power(filter1.reshape(-1), 2))))
            return filter1
        
        for th in thetas:
            gaborKernels[th] = {}
            gaborKernels[th]['0'] = getGaborKernel(gaborparams, th, 0)
            gaborKernels[th]['90'] = getGaborKernel(gaborparams, th, 90)

        return gaborKernels
    
    def calculateFeatureMaps(self, r, g, b, L, params):
        min_rg = np.minimum(r, g)
        b_min_rg = np.abs(np.subtract(b, min_rg))
        CBY = np.divide(b_min_rg, L, out=np.zeros_like(L), where=L != 0)

        r_g = np.abs(np.subtract(r,g))
        CRG = np.divide(r_g, L, out=np.zeros_like(L), where=L != 0)

        colorMaps = {}
        colorMaps['CBY'] = CBY
        colorMaps['CRG'] = CRG
        colorMaps['L'] = L
        
        kernels = self.getGaborKernels(params['gaborparams'], params['thetas'])
        orientationMaps = []
        for th in params['thetas']:
            kernel_0  = kernels[th]['0']
            kernel_90 = kernels[th]['90']
            o1 = cv2.filter2D(L, -1, kernel_0, borderType=cv2.BORDER_REPLICATE)
            o2 = cv2.filter2D(L, -1, kernel_90, borderType=cv2.BORDER_REPLICATE)
            o = np.add(abs(o1), abs(o2))
            orientationMaps.append(o)

        allFeatureMaps = {
            0: colorMaps['CBY'],
            1: colorMaps['CRG'],
            2: colorMaps['L'],
            3: np.array(orientationMaps)
        }
        return allFeatureMaps