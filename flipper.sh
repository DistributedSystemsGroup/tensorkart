for f in samples/*
do
	echo "Flipping data.csv from" $f
	cat $f/data.csv | sed -r 's#png,-#png,x#g' | sed -r 's#png,0#png,-0#g' | sed -r 's#png,1#png,-1#g' | sed -r 's#png,x#png,#g' > $f/tmp.csv; 
	rm $f/data.csv; 
	mv $f/tmp.csv $f/data.csv
done