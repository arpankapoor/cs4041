run: data/data.txt
	@echo "Removing stopwords..."
	@./stage2 data/data.txt
	@echo "Creating vocabulary..."
	@./stage3 data/processed_data.txt
	@echo "Training naive Bayes model..."
	@./stage4 data/processed_data.txt
	@echo "Performing 10-fold cross validation..."
	@./stage5 data/processed_data.txt

data/data.txt:
	@echo "Collecting reviews..."
	@./stage1 data/asins.txt
	@echo "Removing duplicates from the data set..."
	@sort data/data.txt | uniq | shuf > data/tmp
	@mv data/tmp > data/data.txt

clean:
	$(RM) data/data.txt
	$(RM) data/model.p
	$(RM) data/processed_data.txt
	$(RM) data/vocabulary.txt

.PHONY: run clean
