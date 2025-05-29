for i in $(seq 9 27);
do
	alpha=$(echo "scale=2; $i/100" | bc)
	alpha=$(echo 0$alpha)
	for j in $(seq 20 35);
	do
		threshold=$(echo "scale=2; $j/100" | bc)
		threshold=$(echo 0$threshold)
		# python generate_mask.py --anp-alpha=$alpha --output-dir="mask_out/$alpha/"
		python prune_network.py --threshold=$threshold --mask-file="./mask_out/$alpha/mask_values.txt"
	done
done