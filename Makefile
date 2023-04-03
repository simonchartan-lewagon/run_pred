hello:
	echo 'hello world!!'

clean:
	rm -rf models

load_all_models: clean
	python -c "from run_pred.functions.f_training import load_model; load_model('StandardScaler'); load_model('OneHotEncoder'); load_model('StackingRegressor')"

predict:
	python run_pred/interface/main.py
