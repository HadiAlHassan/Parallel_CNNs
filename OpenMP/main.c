#include "lenet.h"
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define FILE_TRAIN_IMAGE	"../train-images-idx3-ubyte"
#define FILE_TRAIN_LABEL	"../train-labels-idx1-ubyte"
#define FILE_TEST_IMAGE		"../t10k-images-idx3-ubyte"
#define FILE_TEST_LABEL		"../t10k-labels-idx1-ubyte"
#define LENET_FILE 			"../model.dat"
#define COUNT_TRAIN		60000
#define COUNT_TEST		10000

//Progress Bar

void display_progress(int current, int total) {
    int barWidth = 50; // Width of the progress bar in characters
    double progress = (double)current / total;

    printf("\r["); // Carriage return and start of the bar
    int pos = barWidth * progress;
    for (int i = 0; i < barWidth; ++i) {
        if (i < pos) printf("#");
        else if (i == pos) printf(">");
        else printf(" ");
    }
    printf("] %d%%", (int)(progress * 100));

    fflush(stdout); // Force output to be displayed
}

int read_data(unsigned char(*data)[28][28], unsigned char label[], const int count, const char data_file[], const char label_file[])
{
    FILE *fp_image = fopen(data_file, "rb");
    FILE *fp_label = fopen(label_file, "rb");
    if (!fp_image||!fp_label) return 1;
	fseek(fp_image, 16, SEEK_SET);
	fseek(fp_label, 8, SEEK_SET);
	fread(data, sizeof(*data)*count, 1, fp_image);
	fread(label,count, 1, fp_label);
	fclose(fp_image);
	fclose(fp_label);
	return 0;
}

void training(LeNet5 *lenet, image *train_data, uint8 *train_label, int batch_size, int total_size)
{
	for (int i = 0, percent = 0; i <= total_size - batch_size; i += batch_size)
	{
		TrainBatch(lenet, train_data + i, train_label + i, batch_size);

		display_progress(i+batch_size, total_size);
	}
	printf(" Training Complete\n"); 

}


int testing(LeNet5 *lenet, image *test_data, uint8 *test_label, int total_size) {
    int right = 0;
    int barUpdateFrequency = total_size / 100; // Adjust the frequency of updates to avoid excessive printing

    #pragma omp parallel for reduction(+:right)
    for (int i = 0; i < total_size; ++i) {
        uint8 l = test_label[i];
        int p = Predict(lenet, test_data[i], 10);
        right += (l == p);

        // Update progress only at regular intervals to avoid excessive printing
        if (i % barUpdateFrequency == 0) {
            #pragma omp critical
            {
                display_progress(i + 1, total_size);
            }
        }
    }

    // Ensure 100% is shown when done
    display_progress(total_size, total_size);
    printf(" Testing Complete\n");
    return right;
}


int save(LeNet5 *lenet, char filename[])
{
	FILE *fp = fopen(filename, "wb");
	if (!fp) return 1;
	fwrite(lenet, sizeof(LeNet5), 1, fp);
	fclose(fp);
	return 0;
}

int load(LeNet5 *lenet, char filename[])
{
	FILE *fp = fopen(filename, "rb");
	if (!fp) return 1;
	fread(lenet, sizeof(LeNet5), 1, fp);
	fclose(fp);
	return 0;
}



void foo()
{
	image *train_data = (image *)calloc(COUNT_TRAIN, sizeof(image));
	uint8 *train_label = (uint8 *)calloc(COUNT_TRAIN, sizeof(uint8));
	image *test_data = (image *)calloc(COUNT_TEST, sizeof(image));
	uint8 *test_label = (uint8 *)calloc(COUNT_TEST, sizeof(uint8));
	if (read_data(train_data, train_label, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL))
	{
		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
		free(train_data);
		free(train_label);
		system("pause");
	}
	if (read_data(test_data, test_label, COUNT_TEST, FILE_TEST_IMAGE, FILE_TEST_LABEL))
	{
		printf("ERROR!!!\nDataset File Not Find!Please Copy Dataset to the Floder Included the exe\n");
		free(test_data);
		free(test_label);
		system("pause");
	}


	LeNet5 *lenet = (LeNet5 *)malloc(sizeof(LeNet5));
	if (load(lenet, LENET_FILE))
		Initial(lenet);
	int batches[] = { 300 };

	double Trainstart_time = omp_get_wtime();

	for (int i = 0; i < sizeof(batches) / sizeof(*batches);++i)
		training(lenet, train_data, train_label, batches[i],COUNT_TRAIN);

	double     Trainend_time = omp_get_wtime();
	double Trainelapsed_time = Trainend_time - Trainstart_time;
	printf("Elapsed time for training: %f seconds\n", Trainelapsed_time);

	double Teststart_time = omp_get_wtime();
	int right = testing(lenet, test_data, test_label, COUNT_TEST);
	double     Testend_time = omp_get_wtime();
	double Testelapsed_time = Testend_time - Teststart_time;
	printf("Elapsed time for testing: %f seconds\n", Testelapsed_time);


	printf("Accuracy: %f %%\n", ((float)right / COUNT_TEST)*100);
	//printf("%d/%d\n", right, COUNT_TEST);
	
	//save(lenet, LENET_FILE);
	free(lenet);
	free(train_data);
	free(train_label);
	free(test_data);
	free(test_label);
	//system("pause");
	printf("Press Enter to continue...");
	getchar();

}


int main()
{		
	double start_time = omp_get_wtime();
	foo();
	double end_time = omp_get_wtime();
	double elapsed_time = end_time - start_time;
    printf("Elapsed time: %f seconds\n", elapsed_time);

	return 0;
}