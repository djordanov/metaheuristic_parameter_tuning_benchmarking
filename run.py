# pylint: disable=no-member

import rpy2.robjects.packages as rpackages

irace = rpackages.importr('irace')
parameters = irace.readParameters('./tuning/sa-parameters.txt')
scenario = irace.readScenario('./tuning/sa-scenario.txt')
irace.checkIraceScenario(scenario = scenario, parameters = parameters)
irace.irace(scenario = scenario, parameters = parameters)