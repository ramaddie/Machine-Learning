package opt.test;

import java.util.Arrays;

import dist.DiscreteDependencyTree;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.DiscreteChangeOneNeighbor;
import opt.EvaluationFunction;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.DiscreteChangeOneMutation;
import opt.ga.SingleCrossOver;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * A test using the flip flop evaluation function
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class FlipFlopTest {
    /** The n value */
    private static final int N = 80;
    
    public static void main(String[] args) {
        int[] ranges = new int[N];
        Arrays.fill(ranges, 2);
        EvaluationFunction ef = new FlipFlopEvaluationFunction();
        Distribution odd = new DiscreteUniformDistribution(ranges);
        NeighborFunction nf = new DiscreteChangeOneNeighbor(ranges);
        MutationFunction mf = new DiscreteChangeOneMutation(ranges);
        CrossoverFunction cf = new SingleCrossOver();
        Distribution df = new DiscreteDependencyTree(.1, ranges); 
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        ProbabilisticOptimizationProblem pop = new GenericProbabilisticOptimizationProblem(ef, odd, df);
        
        // RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);
        // double[] arr = new double[300];

        // for (int i = 0; i < 300; i++) {
        //     FixedIterationTrainer fit = new FixedIterationTrainer(rhc, i);
        //     fit.train();
        //     arr[i] = ef.value(rhc.getOptimal());
        // }
        // for (int j = 0; j < 300; j++) {
        //     System.out.println(arr[j]);
        // }
        //FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
        //fit.train();
        //System.out.println("RHC: " + ef.value(rhc.getOptimal()));
        
        SimulatedAnnealing sa = new SimulatedAnnealing(1E11, .95, hcp);
        double[] arr = new double[300];

        for (int i = 0; i < 300; i++) {
            FixedIterationTrainer fit = new FixedIterationTrainer(sa, i);
            fit.train();
            arr[i] = ef.value(sa.getOptimal());
        }
        for (int j = 0; j < 300; j++) {
            System.out.println(arr[j]);
        }        
        // fit = new FixedIterationTrainer(sa, 200000);
        // fit.train();
        // System.out.println(ef.value(sa.getOptimal()));
        
        // StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 100, 10, gap);
        // double[] arr = new double[300];

        // for (int i = 0; i < 300; i++) {
        //     FixedIterationTrainer fit = new FixedIterationTrainer(ga, i);
        //     fit.train();
        //     arr[i] = ef.value(ga.getOptimal());
        // }
        // for (int j = 0; j < 300; j++) {
        //     System.out.println(arr[j]);
        // }
        // fit = new FixedIterationTrainer(ga, 1000);
        // fit.train();
        // System.out.println(ef.value(ga.getOptimal()));
    }
}
