<script setup>
import { ref, watch } from 'vue';
import axios from 'axios';
import { store } from '../store'; // Import shared store to manage global state

// Reactive references
const uploadedImage = ref(null); // Stores the local URL of the uploaded image
const blurredImage = ref(null); // Stores the URL of the blurred image returned from the backend
const isLoading = ref(false);  // Boolean to show a loading spinner while the image is being processed
const displayedImage = ref(null); // New ref to handle which image is currently displayed
const uploadedFile = ref(null); // Store the uploaded file separately

// Watch for changes in the selected image from the global store
watch(
  () => store.selectedImage,
  (newImage) => {
    if (newImage) {
      console.log("New image selected:", newImage);
      uploadedImage.value = newImage;
      displayedImage.value = newImage; // Set the displayed image
      blurredImage.value = null;
      uploadedFile.value = null; // Clear any previous file reference if selected from gallery
    }
  },
  { immediate: true }
);

const handleImageUpload = (event) => {
  const file = event.target.files[0];
  if (file) {
    const imageUrl = URL.createObjectURL(file);
    uploadedImage.value = imageUrl;
    displayedImage.value = imageUrl; // Update displayed image
    blurredImage.value = null; // Clear the blurred image when a new image is uploaded
    uploadedFile.value = file; // Store the file object
    store.photoUrls.push(imageUrl); // Store the uploaded base photo in the global store
  }
};

// Function to send the uploaded image to the backend and apply a blur effect
const applyBlur = async () => {
  if (!uploadedImage.value) return;

  // Show the loading spinner while the image is being processed
  isLoading.value = true;
  try {
    let formData = new FormData();

    // If there's an uploaded file, use it
    if (uploadedFile.value) {
      formData.append('file', uploadedFile.value);  // Upload the file
    } else {
      // If the image is selected from the gallery (URL), fetch the image data from the URL
      const response = await fetch(uploadedImage.value);
      const blob = await response.blob();
      formData.append('file', blob, 'gallery-image.png');  // Send as a blob
    }

    // Make a POST request to the backend API to apply the blur effect
    const response = await axios.post(`${store.apiUrl}/apply-blur`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
      responseType: 'blob'  // Expect the response as binary data (blob)
    });

    // Create an object URL from the response data (blob) and set it as the blurred image
    const imageUrl = URL.createObjectURL(response.data); // Create a URL for the blob response
    blurredImage.value = imageUrl;
    displayedImage.value = imageUrl;  // Set the displayed image to the blurred version
  } catch (error) {
    console.error('Failed to apply blur:', error);
  } finally {
    isLoading.value = false;
  }
};

// Function to trigger the file input dialog when the image field is clicked
const openFileDialog = () => {
  const fileInput = document.querySelector('input[type="file"]');
  if (fileInput) fileInput.click();
};

// Function to toggle between original and blurred image
const toggleImage = () => {
  if (blurredImage.value && !isLoading.value) {
    displayedImage.value = displayedImage.value === blurredImage.value ? uploadedImage.value : blurredImage.value;
  }
};

// Function to reset the image when "X" button is clicked
const resetImage = () => {
  uploadedImage.value = null;
  blurredImage.value = null;
  displayedImage.value = null;
  uploadedFile.value = null;  // Reset the uploaded file
  const fileInput = document.querySelector('input[type="file"]');
  if (fileInput) {
    fileInput.value = '';  // Clear the file input
  }
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
                  <v-btn v-if="blurredImage" icon density="compact" class="reset-btn ma-2" @click="resetImage"
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
            <v-btn color="primary" @click="applyBlur" :disabled="!uploadedImage || isLoading" block>
              <v-icon left>mdi-upload</v-icon>
              Apply Blur
            </v-btn>
            <!-- Additional instruction text when blurred image is available -->
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
  font-size: 16px;
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