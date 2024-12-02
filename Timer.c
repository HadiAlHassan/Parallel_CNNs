#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main() {
    double AVG = 0;
    int N = 10; // number of trials
    double total_time[N];
	
	for (int k=0; k<N; k++){
		clock_t start_time = clock(); // Start measuring time
		
		// Enter code here (The function call to the cnn must be encapsulated here)
		
		clock_t end_time = clock(); // End measuring time
		
		total_time[k] = ((double)(end_time - start_time)) / CLOCKS_PER_SEC;
		printf("Execution time of trial [%d]: %f seconds\n", i , total_time[k]);
		AVG += total_time[k];
    }
	double avg = AVG/N*100;

    printf("The average execution time of 10 trials is: %f ms", avg);
	
	// Only for parallel implementation from here on:
	double sequential = ; // Enter the value of the sequential time.
	double speedup = sequential / avg;
	double efficiency = (speedup * 100) / n;
	
	printf("The speedup is: %f\nThe efficiency is: %f\n", speedup, efficiency);
	
	return 0;
}