for i in $(seq 1 99);
do
	alpha=$(echo "scale=2; $i/100" | bc)
	alpha=$(echo 0$alpha)
	python generate_mask.py --anp-alpha=$alpha --output-dir="mask_out/$alpha/"
done