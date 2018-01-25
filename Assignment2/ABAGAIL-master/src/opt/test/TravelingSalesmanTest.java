package opt.test;

import java.util.Arrays;
import java.util.Random;

import dist.DiscreteDependencyTree;
import dist.DiscretePermutationDistribution;
import dist.DiscreteUniformDistribution;
import dist.Distribution;

import opt.SwapNeighbor;
import opt.GenericHillClimbingProblem;
import opt.HillClimbingProblem;
import opt.NeighborFunction;
import opt.RandomizedHillClimbing;
import opt.SimulatedAnnealing;
import opt.example.*;
import opt.ga.CrossoverFunction;
import opt.ga.SwapMutation;
import opt.ga.GenericGeneticAlgorithmProblem;
import opt.ga.GeneticAlgorithmProblem;
import opt.ga.MutationFunction;
import opt.ga.StandardGeneticAlgorithm;
import opt.prob.GenericProbabilisticOptimizationProblem;
import opt.prob.MIMIC;
import opt.prob.ProbabilisticOptimizationProblem;
import shared.FixedIterationTrainer;

/**
 * 
 * @author Andrew Guillory gtg008g@mail.gatech.edu
 * @version 1.0
 */
public class TravelingSalesmanTest {
    /** The n value */
    private static final int N = 50;
    /**
     * The test main
     * @param args ignored
     */
    public static void main(String[] args) {
        Random random = new Random();
        // create the random points
        double[][] points = new double[N][2];
        for (int i = 0; i < points.length; i++) {
            points[i][0] = random.nextDouble();
            points[i][1] = random.nextDouble();   
        }
        // for rhc, sa, and ga we use a permutation based encoding
        TravelingSalesmanEvaluationFunction ef = new TravelingSalesmanRouteEvaluationFunction(points);
        Distribution odd = new DiscretePermutationDistribution(N);
        NeighborFunction nf = new SwapNeighbor();
        MutationFunction mf = new SwapMutation();
        CrossoverFunction cf = new TravelingSalesmanCrossOver(ef);
        HillClimbingProblem hcp = new GenericHillClimbingProblem(ef, odd, nf);
        GeneticAlgorithmProblem gap = new GenericGeneticAlgorithmProblem(ef, odd, mf, cf);
        
        RandomizedHillClimbing rhc = new RandomizedHillClimbing(hcp);      
        
        //double[] arr = new double[1000];

        // for (int i = 0; i < 1000; i++) {
        //     FixedIterationTrainer fit = new FixedIterationTrainer(rhc, i);
        //     fit.train();
        //     arr[i] = ef.value(rhc.getOptimal());
        // }
        // for (int j = 0; j < 1000; j++) {
        //     System.out.println(arr[j]);
        // }

        //FixedIterationTrainer fit = new FixedIterationTrainer(rhc, 200000);
        //fit.train();
        //System.out.println(ef.value(rhc.getOptimal()));

        SimulatedAnnealing sa = new SimulatedAnnealing(1E12, .95, hcp);
        // double[] arr = new double[1000];

        // for (int i = 0; i < 1000; i++) {
        //     FixedIterationTrainer fit = new FixedIterationTrainer(sa, i);
        //     fit.train();
        //     arr[i] = ef.value(sa.getOptimal());
        // }
        // for (int j = 0; j < 1000; j++) {
        //     System.out.println(arr[j]);
        // }


        // fit = new FixedIterationTrainer(sa, 200000);
        // fit.train();
        // System.out.println(ef.value(sa.getOptimal()));
        
        StandardGeneticAlgorithm ga = new StandardGeneticAlgorithm(200, 150, 20, gap);
        double[] arr = new double[200];

        for (int i = 0; i < 200; i++) {
            FixedIterationTrainer fit = new FixedIterationTrainer(ga, i);
            fit.train();
            arr[i] = ef.value(ga.getOptimal());
        }
        for (int j = 0; j < 200; j++) {
            System.out.println(arr[j]);
        }
        //fit = new FixedIterationTrainer(ga, 1000);
        //fit.train();
        //System.out.println(ef.value(ga.getOptimal()));


        
    }
}
