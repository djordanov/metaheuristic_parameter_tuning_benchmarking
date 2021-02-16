library(irace)
library(rmake)

setwd('data/irace')
rdata_files = list.files('./', pattern = '*.Rdata')

for (rdata_file in rdata_files) {
  print(rdata_file)

  # check if iterations file (and implicitly test-experiments file) was created already
  if (!file.exists(replaceSuffix(rdata_file, '-iterations.csv'))) {
    print(paste(rdata_file, 'has not been postprocessed yet'))

    # if not, load irace result file and create them...
    load(rdata_file)

    # dump testing results into test experiments file
    ftestExperiments = replaceSuffix(rdata_file, '-test-experiments.csv')
    if (is.null(iraceResults$testing)) {
      print(paste(rdata_file, 'has not been tested yet'))

      # sometimes testing wasn't done for some reason
      iraceResults$scenario$testInstancesDir = paste(iraceResults$scenario$trainInstancesDir, 'test', sep = '/')
      iraceResults$scenario$testInstancesFile = paste(iraceResults$scenario$testInstancesDir, 'testInstancesFile', sep = '/')
      iraceResults$scenario$testNbElites = 1
      iraceResults$scenario$testIterationElites = 1
      
      testing.main(logFile = rdata_file)
      load(rdata_file)
    }
    write.table(iraceResults$testing$experiments, file = ftestExperiments, sep = ",")

    # write iterations with the number of experiments executed into new file
    iterations = unique(iraceResults$experimentLog[,'iteration'])
    cexperiments = c()
    for (iteration in iterations) { 
      cexperiments = c(cexperiments, sum(iraceResults$experimentLog[,'iteration'] == iteration)) 
    }
    elites = iraceResults$iterationElites

    iteration_report = array(c(iterations, cexperiments, elites), dim = c(length(iterations), 3))
    colnames(iteration_report) = c('iteration', 'experiments', 'elite')
    fiterations = replaceSuffix(rdata_file, '-iterations.csv')
    write.table(iteration_report, fiterations, sep = ",")
  }
}