MAX_CLASS_COUNT=15
PREDICT_CLASS_COUNT=9
TOL_BIAS=3
target:
	python3 ./src/build.py $(MAX_CLASS_COUNT) $(PREDICT_CLASS_COUNT) $(TOL_BIAS)
	bash ./src/classify 
clean:
	bash ./src/redo
