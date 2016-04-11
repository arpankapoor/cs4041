run: data/data.txt
	@echo "Removing stopwords..."
	@./stage2 data/data.txt
	@echo "Creating vocabulary..."
	@./stage3 data/processed_data.txt
	@echo "Training naive Bayes model..."
	@./stage4 data/processed_data.txt
	@echo "Performing 10-fold cross validation..."
	@./stage5 data/processed_data.txt
	@echo "Creating vocabulary with bigram features..."
	@./stage6 data/processed_data.txt
	@echo "Training with bigram features and performing 10-fold cross validation..."
	@./stage7 data/processed_data.txt

data/data.txt:
	@echo "Collecting reviews..."
	@./stage1 data/asins.txt
	@echo "Removing duplicates from the data set..."
	@sort -uR data/data.txt -o data/data.txt

clean:
	$(RM) data/data.txt
	$(RM) data/model.p
	$(RM) data/processed_data.txt
	$(RM) data/vocabulary.txt

.PHONY: run clean
