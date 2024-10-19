// Import the reactive and ref functions from Vue
import { reactive, ref } from 'vue';

// Define a reactive store to manage global state across the app
export const store = reactive({
    photoUrls: [], // Stores the URLs of the fetched unblurred photos
    apiUrl: 'http://localhost:8000', // Base URL for API requests
    selectedImage: null, // Stores the selected image
});
