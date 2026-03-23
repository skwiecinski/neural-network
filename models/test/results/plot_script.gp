set title 'Neural Network Training History'
set xlabel 'Epoch'
set ylabel 'Value'
set grid
set key autotitle columnhead
set term pngcairo size 1000,700
set output 'models/test/results/training_history.png'

plot 'models/test/results/training_loss.dat' using 1:2 with lines title 'Average Training Loss' linecolor rgb 'red', \
     'models/test/results/validation_accuracy.dat' using 1:2 with lines title 'Validation Accuracy' linecolor rgb 'blue'
