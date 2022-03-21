import tensorflow as tf
import math, os, glob
import numpy as np
import scipy.io
import ray

class SaliencyMapTf():
    def __init__(self):
        self.params = self.setupParams()
        pass
    
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
    
    def preprocess(self, input_img):
        self.raw_img = input_img
        b= tf.cast(input_img[:,:,0], dtype='float32')
        g= tf.cast(input_img[:,:,1], dtype='float32')
        r= tf.cast(input_img[:,:,2], dtype='float32')
        L= tf.cast(tf.math.maximum(tf.math.maximum(r,g),b), dtype='float32')
        
        b_pyr = self.getPyramids(b, self.params['max_level'])
        g_pyr = self.getPyramids(g, self.params['max_level'])
        r_pyr = self.getPyramids(r, self.params['max_level'])
        L_pyr = self.getPyramids(L, self.params['max_level'])
        
        return [b_pyr, g_pyr, r_pyr, L_pyr]
    
    def compute_saliency(self,input_img):
        [b_pyr, g_pyr, r_pyr, L_pyr] = input_img
        
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

            maps = self.calculateFeatureMaps(p_r, p_g, p_b, p_L, self.params)

            for i in range(0,3):
                resized_m = tf.image.resize(tf.expand_dims(maps[i], axis=-1), (28, 32), method='mitchellcubic')
                featMaps[i].append(tf.squeeze(resized_m, axis=-1))

            for m in maps[3]:
                resized_m = tf.image.resize(tf.expand_dims(m, axis=-1), (28,32),method='mitchellcubic')
                featMaps[3].append(tf.squeeze(resized_m, axis=-1))
                
        activationMaps = []
        activation_sigma = self.params['sigma_frac_act']*tf.math.reduce_mean([32,28]).numpy() # the shape of map
        
        ray.init(ignore_reinit_error=True)
        for i in range(0,4):
            for map in featMaps[i]:
                activationMaps.append(self.calculate(map, activation_sigma))
        
        normalisedActivationMaps = []
        normalisation_sigma = self.params['sigma_frac_norm']*tf.math.reduce_mean([32,28]).numpy()
        
        for map in activationMaps:
            normalisedActivationMaps.append(self.normalize.remote(map, normalisation_sigma))
        
        mastermap = normalisedActivationMaps[0]
        for i in range(1, len(normalisedActivationMaps)):
            mastermap = tf.math.add(normalisedActivationMaps[i], mastermap)

        mastermap_res = tf.image.resize(tf.expand_dims(mastermap, axis=-1), (self.raw_img.shape[0], self.raw_img.shape[1]), method='bicubic')

        return mastermap_res*255.0
    
    def loadGraphDistanceMatrixFor28x32(self):
        f = scipy.io.loadmat("./resources/28__32__m__2.mat")
        return tf.convert_to_tensor(f['grframe'][0][0][0])
    
    def stm_normalize(self, state_transition_matrix, axis=0, norm='l1'):
        X = tf.transpose(state_transition_matrix)
        norms = tf.math.reduce_sum(tf.math.abs(X), axis=1)
        X /= norms[..., tf.newaxis]
        return tf.transpose(X)
    
    @ray.remote
    def calculate(self, map, sigma):
        distanceMat = self.loadGraphDistanceMatrixFor28x32()
        denom = 2 * pow(sigma, 2)

        expr = -tf.math.divide(tf.convert_to_tensor(distanceMat), denom)
        Fab = tf.math.exp(expr)
        
        map_linear = tf.reshape(tf.transpose(map), -1) # column major
        state_transition_matrix = Fab * tf.transpose(tf.math.abs(
            tf.transpose(tf.zeros((distanceMat.shape[0], distanceMat.shape[1])) + map_linear) - map_linear
        ))
        
        norm_STM = self.stm_normalize(state_transition_matrix, axis=0, norm='l1')
        eVec = self.markovchain(norm_STM, 0.0001)
        processed_reshaped = tf.transpose(tf.reshape(eVec, (map.shape[1],map.shape[0])))

        return processed_reshaped
    
    @ray.remote
    def normalize(self, map, sigma):
        distanceMat= self.loadGraphDistanceMatrixFor28x32()
        denom = 2 * pow(sigma, 2)
        expr = -tf.math.divide(tf.convert_to_tensor(distanceMat), denom)
        Fab = tf.math.exp(expr)


        map_linear = tf.reshape(tf.transpose(map), -1)
        state_transition_matrix = tf.transpose(tf.transpose(Fab) * tf.math.abs(map_linear))

        norm_STM = self.stm_normalize(state_transition_matrix, axis=0, norm='l1')
        eVec = self.markovchain(norm_STM, 0.0001)
        processed_reshaped = tf.transpose(tf.reshape(eVec, (map.shape[1], map.shape[0])))

        return processed_reshaped
    
    def markovchain(self, mat, tolerance):
        mat = tf.cast(mat, dtype='float32')
        w,h = mat.shape
        diff = 1
        v = tf.math.divide(tf.ones((w, 1), dtype='float32'), w)
        oldv = v
        oldoldv = v
        while diff > tolerance :
            oldv = v
            oldoldv = oldv
            v = tf.reduce_sum(mat[...,tf.newaxis]* v, axis=-2)
            diff = tf.linalg.normalize(tf.squeeze(oldv - v,axis=-1), ord=2)[1]
            s = sum(v)
            if s>=0 and s< np.inf:
                continue
            else:
                v = oldoldv
                break

        v = tf.math.divide(v, sum(v))

        return v
                
    def tf_pyr(self, img):
        img = tf.squeeze(
            tf.image.resize(
                tf.expand_dims(img,axis=-1),
                [round(img.shape[0]/2),round(img.shape[1]/2)],
                method='nearest')
            , axis= -1)
        return img
    
    def getPyramids(self, image, max_level):
        imagePyr = [self.tf_pyr(image)]
        for i in range(1, max_level):
            imagePyr.append(self.tf_pyr(imagePyr[i-1]))
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

            psi = tf.constant(math.pi) /180 *phase
            rtDeg = tf.constant(math.pi) /180 *angle

            omega = 2 * tf.constant(math.pi) / gp['filterPeriod']
            co = math.cos(rtDeg)
            si = -math.sin(rtDeg)
            major_sigq = 2 * pow(major_sd, 2)
            minor_sigq = 2 * pow(minor_sd, 2)

            vec = range(-int(sz), int(sz) + 1)
            vlen = len(vec)
            vco = [i * co for i in vec]
            vsi = [i * si for i in vec]
            a = tf.transpose(tf.repeat(tf.expand_dims(tf.convert_to_tensor(vco),axis=0),vlen, axis = 0))
            b = tf.repeat(tf.expand_dims(vsi, axis=0), vlen, axis=0)
            major = a + b
            major2 = tf.math.pow(major, 2)

            a = tf.transpose(tf.repeat(tf.expand_dims(tf.convert_to_tensor(vsi),axis=0),vlen, axis = 0))
            b = tf.repeat(tf.expand_dims(vco, axis=0), vlen, axis= 0)
            minor = a + b
            minor2 = tf.math.pow(minor ,2)

            a = tf.math.cos(omega * major + psi)
            b = tf.math.exp(-major2 / major_sigq - minor2 / minor_sigq)
            result = tf.math.multiply(a, b)

            filter1 = tf.math.subtract(result, tf.math.reduce_mean(tf.reshape(result, -1)))
            filter1 = tf.math.divide(filter1, tf.math.sqrt(tf.math.reduce_sum(tf.math.pow(tf.reshape(filter1, -1),2))))
            return filter1
        
        for th in thetas:
            gaborKernels[th] = {}
            gaborKernels[th]['0'] = getGaborKernel(gaborparams, th, 0)
            gaborKernels[th]['90'] = getGaborKernel(gaborparams, th, 90)

        return gaborKernels
        
    
    def calculateFeatureMaps(self, r, g, b, L, params):
        min_rg = tf.math.minimum(r, g)
        b_min_rg = tf.math.abs(tf.math.subtract(b, min_rg))
        CBY = tf.clip_by_value(tf.math.divide_no_nan(b_min_rg, L), 0, 1)

        r_g = np.abs(np.subtract(r,g))
        r_g = tf.math.abs(tf.math.subtract(r, g))
        CRG = tf.clip_by_value(tf.math.divide_no_nan(r_g, L),0,1)

        colorMaps = {}
        colorMaps['CBY'] = CBY
        colorMaps['CRG'] = CRG
        colorMaps['L'] = L
        
        kernels = self.getGaborKernels(params['gaborparams'], params['thetas'])
        orientationMaps = []

        for th in params['thetas']:
            kernel_0  = kernels[th]['0']
            kernel_90 = kernels[th]['90']
            np_L = np.asarray(L)
            np_kernel_0 = np.asarray(kernel_0)
            np_kernel_90 = np.asarray(kernel_90)

            o1 = tf.nn.conv2d(
                tf.expand_dims(tf.expand_dims(L,axis=0),axis=-1),
                tf.expand_dims(tf.expand_dims(kernel_0, axis=-1),axis=-1),
                strides = [1,1,1,1],
                padding='SAME')
            o1 = tf.squeeze(o1, [0,-1])
            o2 = tf.nn.conv2d(
                tf.expand_dims(tf.expand_dims(L,axis=0),axis=-1),
                tf.expand_dims(tf.expand_dims(kernel_90, axis=-1),axis=-1),
                strides = [1,1,1,1],
                padding='SAME')
            o2 = tf.squeeze(o2, [0,-1])
            o = tf.math.add(abs(o1), abs(o2))
            orientationMaps.append(o)
            
        allFeatureMaps = {
            0: colorMaps['CBY'],
            1: colorMaps['CRG'],
            2: colorMaps['L'],
            3: orientationMaps
        }
        return allFeatureMaps