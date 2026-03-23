set title 'Neural Network Training History'
set xlabel 'Epoch'
set ylabel 'Value'
set grid
set key autotitle columnhead
set term pngcairo size 1000,700
set output 'models/root/results/training_history.png'

plot 'models/root/results/training_loss.dat' using 1:2 with lines title 'Average Training Loss' linecolor rgb 'red', \
     'models/root/results/validation_accuracy.dat' using 1:2 with lines title 'Validation Accuracy' linecolor rgb 'blue'
