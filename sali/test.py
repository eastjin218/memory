import os, glob
from saliencymap  import SaliencyMap
from matplotlib import pyplot as plt

f_path = glob.glob('Itti/src/gbvs/images/0_*')
sali = SaliencyMap(f_path[0])
result = sali.compute_saliency()
