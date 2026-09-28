# 1
再简单写完v1,的时候，train的时候一切正常，但是inference的时候直接loss爆炸，发现是action直接发散了： gpt给出的是 多步推理会累积模型误差  
solution1： 改noise_schedule v1 是线性的，A99的噪声都不算很高