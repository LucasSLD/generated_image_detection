tip_res_cache_size = """accuracy on taskA with cache of size 2: 0.6822420358657837
accuracy on testMeta with cache of size 2: 0.6530447006225586
accuracy on taskA with cache of size 4: 0.692460298538208
accuracy on testMeta with cache of size 4: 0.7086349129676819
accuracy on taskA with cache of size 6: 0.7045634984970093
accuracy on testMeta with cache of size 6: 0.7354629635810852
accuracy on taskA with cache of size 8: 0.7547619342803955
accuracy on testMeta with cache of size 8: 0.7449463605880737
accuracy on taskA with cache of size 16: 0.7176587581634521
accuracy on testMeta with cache of size 16: 0.7693411707878113
accuracy on taskA with cache of size 32: 0.6936507821083069
accuracy on testMeta with cache of size 32: 0.7455078363418579
accuracy on taskA with cache of size 64: 0.6016865372657776
accuracy on testMeta with cache of size 64: 0.7120040059089661
accuracy on taskA with cache of size 128: 0.6962301731109619
accuracy on testMeta with cache of size 128: 0.7583603858947754
accuracy on taskA with cache of size 200: 0.6840277910232544
accuracy on testMeta with cache of size 200: 0.7576740980148315
accuracy on taskA with cache of size 1000: 0.6862103343009949
accuracy on testMeta with cache of size 1000: 0.751185417175293
accuracy on taskA with cache of size 2000: 0.6839285492897034
accuracy on testMeta with cache of size 2000: 0.7528699636459351 
"""
tip_res_cache_size = tip_res_cache_size.split("\n")
d = {"taskA":{str(2*size): 0 for i, size in enumerate((1,2,3,4,8,16,32,64,100,500,1000))},
     "testMeta":{str(2*size): 0 for i, size in enumerate((1,2,3,4,8,16,32,64,100,500,1000))}}

for i, size in enumerate((1,2,3,4,8,16,32,64,100,500,1000)):
    d["taskA"] = 