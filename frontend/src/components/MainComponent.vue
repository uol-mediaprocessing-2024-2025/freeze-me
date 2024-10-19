<script setup>
import { ref, watch } from 'vue';
import axios from 'axios';
import { store } from '../store'; // Import shared store to manage global state

// Reactive references
const fileBlob = ref(null); // Store the image as a Blob (either from upload or gallery)
const blurredImage = ref(null); // Stores the blurred image from the backend
const isLoading = ref(false);  // Boolean to show a loading spinner while the image is being processed
const displayedImage = ref(null); // Handles the image currently displayed (original/blurred)

// Watch for changes in the selected image from the gallery
watch(
  () => store.selectedImage,
  async (newImage) => {
    if (newImage) {
      console.log("New image selected from gallery:", newImage);
      const response = await fetch(newImage);
      fileBlob.value = await response.blob();  // Convert gallery image to blob
      displayedImage.value = newImage; // Display the selected image
      blurredImage.value = null;
    }
  },
  { immediate: true }
);

// Handle image upload
const handleImageUpload = (event) => {
  const file = event.target.files[0];
  if (file) {
    fileBlob.value = file; // Store the uploaded file as a Blob
    displayedImage.value = URL.createObjectURL(file); // Display the uploaded image
    blurredImage.value = null;
    store.photoUrls.push(displayedImage.value); // Store the uploaded image in the global store
  }
};

// Send the uploaded image to the backend and apply a blur effect
const applyBlur = async () => {
  if (!fileBlob.value) return;

  // Show the loading spinner while the image is being processed
  isLoading.value = true;
  try {
    const formData = new FormData();
    formData.append('file', fileBlob.value);  // Send the blob

    // Make a POST request to the backend API to apply the blur effect
    const response = await axios.post(`${store.apiUrl}/apply-blur`, formData, {
      responseType: 'blob'  // Expect binary data (blob)
    });

    // Update the blurred image
    blurredImage.value = URL.createObjectURL(response.data);
    displayedImage.value = blurredImage.value;
  } catch (error) {
    console.error('Failed to apply blur:', error);
  } finally {
    isLoading.value = false;
  }
};

// Trigger the file input dialog when the image field is clicked
const openFileDialog = () => document.querySelector('input[type="file"]').click();

// Toggle between original and blurred image
const toggleImage = () => {
  if (blurredImage.value && !isLoading.value) {
    displayedImage.value = displayedImage.value === blurredImage.value ? URL.createObjectURL(fileBlob.value) : blurredImage.value;
  }
};

// Reset image when 'X' is clicked
const resetImage = () => {
  fileBlob.value = null;
  blurredImage.value = null;
  displayedImage.value = null;
  document.querySelector('input[type="file"]').value = '';
};
</script>

<template>
  <!-- Main container to center the content on the screen -->
  <v-container class="d-flex flex-column align-center justify-center main-container">
    <!-- A card to contain the form and images -->
    <v-card elevation="2" class="pa-4 card-container">
      <!-- Card title -->
      <v-card-title class="justify-center">
        <h2>Image Blur</h2>
      </v-card-title>
      <!-- Card content -->
      <v-card-text>
        <!-- Row for image upload and button -->
        <v-row align="center">
          <!-- Image upload field (clickable image area) -->
          <v-col cols="12" md="8">
            <!-- Wrapper div for positioning the loading overlay -->
            <div class="image-wrapper">
              <v-responsive @click="openFileDialog" class="image-placeholder">
                <!-- Display current image (original or blurred) or placeholder -->
                <v-img v-if="displayedImage" :src="displayedImage" max-height="300" contain @click.stop="toggleImage"
                  :class="{ 'clickable': blurredImage && !isLoading }">
                  <!-- "X" button to reset the image -->
                  <v-btn v-if="displayedImage" icon density="compact" class="reset-btn ma-2" @click="resetImage"
                    color="red">
                    <v-icon small>mdi-close</v-icon>
                  </v-btn>
                </v-img>
                <div class="d-flex align-center justify-center" v-else>Click to upload an image</div>
              </v-responsive>
              <!-- Loading overlay with centered spinner -->
              <div v-if="isLoading" class="loading-overlay">
                <v-progress-circular indeterminate color="primary" size="50"></v-progress-circular>
              </div>
            </div>
          </v-col>
          <!-- Apply Blur Button with Icon -->
          <v-col cols="12" md="3" class="text-center">
            <v-btn color="primary" @click="applyBlur" :disabled="!fileBlob || isLoading" block>
              <v-icon left>mdi-upload</v-icon>
              Apply Blur
            </v-btn>
            <div v-if="blurredImage && !isLoading" class="mt-2 text-caption">
              Click the image to toggle between original and blurred versions
            </div>
          </v-col>
        </v-row>
        <!-- File input field (hidden) -->
        <v-file-input label="Upload an Image" @change="handleImageUpload" accept="image/*" class="d-none"
          prepend-icon="mdi-upload"></v-file-input>
      </v-card-text>
    </v-card>
  </v-container>
</template>

<style scoped>
.main-container {
  height: 100vh;
}

.card-container {
  max-width: 800px;
  width: 100%;
}

.image-wrapper {
  position: relative;
}

.image-placeholder {
  cursor: pointer;
  height: 300px;
  background-color: #f5f5f5;
  border: 1px dashed #ccc;
  border-radius: 8px;
  display: flex;
  justify-content: center;
  align-items: center;
  color: #aaa;
}

.loading-overlay {
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  bottom: 0;
  background-color: rgba(255, 255, 255, 0.7);
  display: flex;
  justify-content: center;
  align-items: center;
  border-radius: 8px;
}

.reset-btn {
  position: absolute;
  top: 6px;
  right: 6px;
}

.clickable {
  cursor: pointer;
}

@media (max-width: 768px) {
  .image-placeholder {
    height: 200px;
  }
}
</style>