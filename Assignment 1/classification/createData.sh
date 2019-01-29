#!/bin/bash
rm perceptron1_train.csv perceptron1_test.csv
for i in {1..10}
do
	num=`expr 100 \* $i`
	python dataClassifier.py -c 1vr -t $num
done