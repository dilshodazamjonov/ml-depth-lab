# TinyTorch Data Loading — Session Plan

## Objective

Build a small data-loading system that converts stored data into batches that can later be passed into a model.

By the end of this session, I should understand the full pipeline:

**Storage → Dataset → Individual samples → DataLoader → Batches → Model**

The goal is not just to finish four classes. I should understand why data storage, sample retrieval, shuffling, and batching are separate responsibilities.

---

## Components to Build

- [ ] `Dataset`
- [ ] `TensorDataset`
- [ ] `DataLoader`
- [ ] `ImageLoader` / image-specific dataset

> Naming decision: determine whether `ImageLoader` is actually responsible for loading one image at a time. If so, `ImageDataset` may be a clearer name because `DataLoader` already handles batch loading.

---

## 1. Review Tensor Requirements

Before building the data classes, verify that my existing `Tensor_CP` supports the operations they will depend on.

- [ ] Check that `len(tensor)` represents the size of dimension `0`
- [ ] Check that `tensor[index]` returns a `Tensor_CP`
- [ ] Check scalar indexing behaviour
- [ ] Check whether tensors can be stacked along a new dimension
- [ ] Decide whether stacking belongs in `Tensor_CP` or inside `DataLoader`
- [ ] Confirm that tensor shapes are preserved correctly after indexing

---

## 2. Build the `Dataset` Abstraction

### Purpose

`Dataset` defines the minimum interface required to represent any collection of samples.

It should answer two questions:

1. How many samples exist?
2. What sample exists at a given index?

### Tasks

- [ ] Create `Dataset` as an abstract base class
- [ ] Define the required length operation
- [ ] Define the required indexing operation
- [ ] Ensure `Dataset` cannot be instantiated directly
- [ ] Ensure subclasses are required to implement both operations

### Questions to Answer

- [ ] Why should `Dataset` not know anything about batching?
- [ ] Why should it not know whether its data comes from tensors, images, files, or a database?
- [ ] How does this small interface allow different datasets to work with the same `DataLoader`?

### Completion Criteria

- [ ] I can explain the difference between a dataset and a data loader
- [ ] I can create a minimal test dataset that follows the interface
- [ ] Instantiating an incomplete dataset subclass fails clearly

---

## 3. Build `TensorDataset`

### Purpose

`TensorDataset` wraps one or more tensors that are already stored in memory.

Every tensor must represent the same number of samples along dimension `0`.

Example concept:

- Feature tensor: `(number_of_samples, number_of_features)`
- Label tensor: `(number_of_samples,)`
- One returned sample: `(feature_at_index, label_at_index)`

### Tasks

- [ ] Accept one or more tensors
- [ ] Validate that at least one tensor was provided
- [ ] Validate that every input is a `Tensor_CP`
- [ ] Validate that all tensors have equal length along dimension `0`
- [ ] Return the number of samples
- [ ] Return aligned tensor elements for a requested index
- [ ] Return each sample as a tuple
- [ ] Preserve the original tensor order

### Design Decisions

- [ ] Decide how zero-dimensional tensors should be handled
- [ ] Decide whether negative indexing is supported
- [ ] Decide what happens for an out-of-range index
- [ ] Decide what happens when tensors contain zero samples
- [ ] Keep feature-label alignment intact

### Tests

- [ ] Dataset with one tensor
- [ ] Dataset with features and labels
- [ ] Dataset with more than two aligned tensors
- [ ] Mismatched first dimensions
- [ ] Non-tensor input
- [ ] No tensors provided
- [ ] Empty tensors
- [ ] First and last valid indices
- [ ] Invalid index

### Completion Criteria

- [ ] Dataset length matches the number of samples
- [ ] Each returned sample contains correctly aligned tensor slices
- [ ] Returned values remain `Tensor_CP` objects
- [ ] Invalid input produces understandable errors

---

## 4. Design the `DataLoader`

### Purpose

`DataLoader` controls:

1. The order in which samples are accessed
2. How samples are divided into batches
3. How individual samples are combined into batched tensors

It should work with any object that follows the `Dataset` interface.

### Constructor Responsibilities

- [ ] Accept a dataset
- [ ] Accept a batch size
- [ ] Accept a shuffle option
- [ ] Validate the dataset
- [ ] Validate that batch size is a positive integer
- [ ] Store configuration without immediately loading all samples

### Batch Count

- [ ] Calculate the total number of batches
- [ ] Include the final incomplete batch
- [ ] Return zero batches for an empty dataset
- [ ] Verify behaviour when batch size is larger than the dataset

The current design does not include `drop_last`, so the final smaller batch should be preserved.

---

## 5. Implement the DataLoader Iteration Model

### Expected Behaviour

Every new iteration over the `DataLoader` represents a new epoch.

At the beginning of each epoch:

- Create the sample indices
- Shuffle the indices if requested
- Divide them into batch-sized groups
- Retrieve only the samples required for the current batch
- Yield one batch at a time

### Tasks

- [ ] Create indices without modifying the original dataset
- [ ] Preserve sequential order when shuffling is disabled
- [ ] Shuffle indices when shuffling is enabled
- [ ] Perform a fresh shuffle for every new epoch
- [ ] Retrieve samples lazily
- [ ] Yield batches one at a time
- [ ] Include the final incomplete batch
- [ ] Preserve every sample’s feature-label relationship

### Important Invariant

If a sample contains a feature and its corresponding label, shuffling must move the complete sample. Features and labels must never be shuffled independently.

### Questions to Answer

- [ ] Why do we shuffle indices instead of the underlying tensors?
- [ ] Why should shuffling happen when iteration begins?
- [ ] Why does lazy iteration use less working memory?
- [ ] What state needs to survive between consecutive batches?

---

## 6. Build Batch Collation

### Purpose

Collation converts a list of individual samples into a tuple of batch tensors.

If every sample contains:

- an input tensor;
- a label tensor;

then collation should produce:

- one stacked input batch;
- one stacked label batch.

A new batch dimension should be introduced at axis `0`.

### Tasks

- [ ] Handle an empty batch deliberately
- [ ] Determine how many tensor fields each sample contains
- [ ] Confirm that all samples have the same number of fields
- [ ] Group sample tensors by their position
- [ ] Stack tensors belonging to the same field
- [ ] Return the batch as a tuple of `Tensor_CP` objects

### Shape Reasoning Exercises

Before implementation, predict the results manually:

- [ ] Several feature vectors become a feature matrix
- [ ] Several scalar labels become a label vector
- [ ] Several images become a four-dimensional image batch
- [ ] The final incomplete batch has the correct first dimension

### Edge Cases

- [ ] Samples contain different numbers of tensors
- [ ] Corresponding tensors have incompatible shapes
- [ ] Batch contains only one sample
- [ ] Dataset returns something other than the expected sample tuple

### Completion Criteria

- [ ] Batch dimension is added correctly
- [ ] Sample order inside each batch is preserved
- [ ] Every output is a `Tensor_CP`
- [ ] Shape incompatibilities fail clearly

---

## 7. Test the Complete Tensor Pipeline

Create a small conceptual dataset with known values so that order and alignment can be checked manually.

### Tests Without Shuffling

- [ ] Iterate through every batch
- [ ] Confirm sequential sample order
- [ ] Confirm the expected number of batches
- [ ] Confirm full batch shapes
- [ ] Confirm the final partial batch shape
- [ ] Confirm that every sample appears exactly once

### Tests With Shuffling

- [ ] Confirm that order changes
- [ ] Confirm that features remain paired with their labels
- [ ] Confirm that every sample still appears exactly once
- [ ] Confirm that a new epoch creates a new ordering
- [ ] Avoid tests that depend on one exact random permutation

### Additional Tests

- [ ] Batch size of `1`
- [ ] Batch size equal to dataset size
- [ ] Batch size larger than dataset size
- [ ] Dataset size not divisible by batch size
- [ ] Empty dataset
- [ ] Invalid batch sizes

---

## 8. Design the Image-Specific Component

### Purpose

The image component represents a disk-backed dataset.

It should initially store lightweight metadata such as:

- image paths;
- corresponding labels;
- an optional transformation pipeline.

It should not load every image during construction.

### Expected Retrieval Process

When one index is requested:

1. Find the corresponding image path
2. Read that image from disk
3. Decode it into an array
4. Convert it into the selected image format
5. Apply optional transformations
6. Convert it into `Tensor_CP`
7. Return the image and its label as one sample

### Tasks

- [ ] Decide whether the class should be called `ImageLoader` or `ImageDataset`
- [ ] Store aligned image paths and labels
- [ ] Validate that paths and labels have equal lengths
- [ ] Report the number of images
- [ ] Load only the requested image
- [ ] Convert every image to a consistent colour mode
- [ ] Convert the image array into a consistent shape
- [ ] Convert pixels into the chosen data type and range
- [ ] Convert the label into a tensor
- [ ] Apply an optional transform before returning the sample

---

## 9. Define the Image Convention

Document one consistent format before implementing image loading.

### Recommended Initial Convention

- [ ] Colour mode: RGB
- [ ] Tensor layout: `(channels, height, width)`
- [ ] Batched layout: `(batch, channels, height, width)`
- [ ] Data type: floating point
- [ ] Pixel range: `[0, 1]`
- [ ] Labels: class indices represented as tensors
- [ ] Loading strategy: lazy, one requested image at a time

### Decisions to Make

- [ ] How grayscale images are handled
- [ ] How images with an alpha channel are handled
- [ ] Whether images of different sizes are rejected or resized
- [ ] What happens when a file does not exist
- [ ] What happens when an image is corrupted
- [ ] Whether transformations receive arrays or tensors

---

## 10. Test the Image Pipeline

Use a tiny image collection rather than a real large dataset.

### Tests

- [ ] Correct dataset length
- [ ] Correct path-label alignment
- [ ] One image loads successfully
- [ ] Returned image is a `Tensor_CP`
- [ ] Returned label is a `Tensor_CP`
- [ ] Image shape follows the selected convention
- [ ] Pixel values follow the selected range
- [ ] RGB conversion is consistent
- [ ] Missing file fails clearly
- [ ] Corrupted image fails clearly
- [ ] Transform is applied when provided
- [ ] No image is loaded until it is requested

### DataLoader Integration

- [ ] Pass the image dataset into the same `DataLoader`
- [ ] Produce a valid image batch
- [ ] Check the complete batch shape
- [ ] Check the final partial batch
- [ ] Confirm image-label alignment after shuffling

---

## 11. Document Responsibilities

After implementation, I should be able to explain each component in one sentence.

- [ ] `Dataset`: defines how to access one sample
- [ ] `TensorDataset`: retrieves aligned samples from in-memory tensors
- [ ] Image dataset: retrieves and prepares one image from disk
- [ ] `DataLoader`: selects, orders, groups, and collates samples
- [ ] Transform: modifies one sample before batching
- [ ] Training loop: consumes batches and performs learning

---

## 12. Final Acceptance Checklist

The session is complete when:

- [ ] `Dataset` defines and enforces the common interface
- [ ] `TensorDataset` returns correctly aligned tensor samples
- [ ] `DataLoader` works with any valid dataset
- [ ] Batch count is correct
- [ ] Full and partial batches are handled correctly
- [ ] Shuffling occurs independently for each epoch
- [ ] Shuffling never breaks feature-label alignment
- [ ] Batch collation produces the correct shapes
- [ ] Images are loaded lazily from disk
- [ ] Image representation is consistent
- [ ] The same `DataLoader` works with tensor and image datasets
- [ ] Important invalid inputs have tests
- [ ] I can explain the architecture without looking at the implementation

---

## Out of Scope for This Session

Do not add these features yet:

- Multiprocessing workers
- Background prefetching
- GPU transfer
- Pinned memory
- Distributed sampling
- Weighted sampling
- Dataset caching
- Advanced image augmentation
- Automatic performance optimization
- Infinite or streaming datasets
- A general-purpose collation system for arbitrary Python objects

These can be added later after the basic abstraction is correct.

---

## Core Learning Outcome

The most important idea to understand is:

> A dataset defines how to retrieve one sample. A data loader decides which samples to retrieve and combines them into batches.

If this separation is clear and the same `DataLoader` works with both tensor-backed and image-backed datasets, the session has achieved its purpose.