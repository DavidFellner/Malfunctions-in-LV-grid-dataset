    Set settings for QDS (quasi dynamic load flow simulation in a power grid) and elements and save results to file to
    create a data set executing a QDS.
    At first the grid is prepared and scenario settings are set. Then samples are created from raw data.
    These samples are time series of voltages at points of connection of households and photovoltaics (PVs) of a low
    voltage distribution network. Finally a deep learning approach is compared to a linear classifier to either
    determine if a sample is from a term with PV (class 1) or no PV (class 0) or from a term with a regularly
    behaving PV (class 0) or a PV with a malfunctioning reactive power control curve (class 1).
    Additionally a dummy dataset can be created that only consists of samples that are constant over the entire
    timeseries (class 0) and samples that are not (class 1). Randomly chosen samples of either classes are plotted
    along with execution at default.
    See framework diagrams for a better overview.

    Choose experiment (dataset and learning settings) in experiment_config.py
    Predefined experiments vary the dataset type (dummy, PV vs no PV, regular PV vs malfunctioning PV) as well as the
    timeseries length of samples (1 day vs 1 week) and the number of samples (too little, 'just enough', sufficient to
    produce a meaningful output after training with the basic network design used, i.e. no Fscore ill defined because only
    always one class predicted in any run of cross validation; note that 1 day vs 7 days also means increasing the amount
    of data points, therefore redundant experiments (i.e. increasing the sample number even more for 1 day timeseries
    experiments was neglected to allow for a better orientation between experiments)
    The experiment also defines the network architecture (in the predefined experiments this is a simple 2 layer Elman
    RNN with 6 hidden nodes in each layer). Multiple options are available such as changing the mini batch size, early
    stopping, warm up, controlling the learning rate...

    Metrics: Deep learning approach should perform better than linear classifier (which just guesses between 0 and 1 class)
             meaning that a higher Fscore should be achieved
             Experiment configs state if this goal can be fulfilled with the experiment settings

                    Task      Dataset collection  ANN design  ANN tuning  Results     Report      Presentation
      Time planned: (Hours)   15                  7.5         15          7.5         10          4
      Time spent:   (Hours)   ~15                 25          ~15         5             to be seen
      Conclusion:   It took much longer than planned to actually get the RNN running and producing meaningful outputs