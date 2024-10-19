<script setup>
import { onMounted, ref } from 'vue'; // Import lifecycle hook and ref from Vue
import { store } from '../store';// Import the global store to access shared state
import { useRouter } from 'vue-router'; // Import Vue Router for navigation

// Reactive reference to hold the list of images
const images = ref([]);
const router = useRouter(); // Initialize the router

// onMounted lifecycle hook to fetch images when the component is mounted
onMounted(() => {
    images.value = store.photoUrls;
});

// Function to handle image click and navigate to the main view
const handleImageClick = (imageSrc) => {
    store.selectedImage = imageSrc; // Set the selected image in the store
    router.push('/'); // Navigate to the main view
};
</script>

<template>
    <!-- Gallery container -->
    <div class="gallery px-4 py-4">
        <!-- Vuetify grid to organize images -->
        <v-row dense>
            <!-- Loop through the images array and display each image -->
            <v-col v-for="(imgSrc, index) in images" :key="index" class="d-flex child-flex" cols="12" sm="6" md="4"
                lg="3" xl="2">
                <!-- Display each image with Vuetify's v-img component -->
                <v-img :src="imgSrc" aspect-ratio="1.67" class="mb-4 clickable-image" @click="handleImageClick(imgSrc)">
                    <!-- Show a loading spinner while the image is loading -->
                    <template v-slot:placeholder>
                        <v-row align="center" class="fill-height ma-0" justify="center">
                            <v-progress-circular color="grey-lighten-5" indeterminate></v-progress-circular>
                        </v-row>
                    </template>
                </v-img>
            </v-col>
        </v-row>
    </div>
</template>

<style scoped>
.gallery {
    display: flex;
    flex-wrap: wrap;
    justify-content: center;
}

.v-img {
    border-radius: 12px;
    cursor: pointer;
    transition: transform 0.3s ease-in-out;
    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
}

.v-img:hover {
    transform: scale(1.1);
    box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
}
</style>
