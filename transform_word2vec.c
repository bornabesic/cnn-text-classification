#include <stdio.h>

/*
	Splits GoogleNews-vectors-negative300.bin into two files:
		word2vec_indices
		word2vec_vectors
*/
int main(void){

	FILE *file = fopen("GoogleNews-vectors-negative300.bin", "rb");
	if(file==NULL){
		printf("Cannot open GoogleNews-vectors-negative300.bin\n");
		return 1;
	}
	printf("float size: %d\n", sizeof(float));

	FILE *indices_file = fopen("word2vec_indices", "wt");
	FILE *vectors_file = fopen("word2vec_vectors", "wb");

	long long num_words, word_dim;

	fscanf(file, "%lld", &num_words);
	fscanf(file, "%lld", &word_dim);

	printf("Number of words: %lld\n", num_words);

	int i;
	char string_buffer[4096];
	float vector_buffer[word_dim];
	for (i=0; i<num_words; i++) {

		fscanf(file, "%s%*c", string_buffer);
		fprintf(indices_file, "%s\t%d\n", string_buffer, i);

		fread(&vector_buffer, sizeof(float), word_dim, file);
		fwrite(&vector_buffer, sizeof(float), word_dim, vectors_file);
	}

	fclose(indices_file);
	fclose(vectors_file);

	return 0;
}