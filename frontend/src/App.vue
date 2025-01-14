<template>
    <!-- Main application container -->
    <v-app id="app">
      <div style="background-color: rgb(30, 50, 92); height: 7vh" class="px-2 text-white d-flex align-center header">
        <div class="pl-4 pr-12 title" >
          <v-app-bar-title>Freeze Me</v-app-bar-title>
        </div>
        <!-- Link to the main view -->
        <v-btn class="d-flex mr-4 no-focus-highlight" variant="outlined" prepend-icon="mdi-home" link to="/">
            Create New Project
        </v-btn>

        <!-- Link to the gallery view -->
        <v-btn class="d-flex no-focus-highlight" variant="outlined" prepend-icon="mdi-image" link to="/gallery">
            Projects
        </v-btn>
      </div>

      <!-- Main content area for rendering views -->
      <v-main style="height: 93vh; overflow-x: hidden;">
          <RouterView /> <!-- Dynamically renders the component based on the route -->
      </v-main>
    </v-app>
</template>

<script>
// Import the Vue Router components for rendering dynamic views
import {
    RouterLink, // Component for linking to different routes
    RouterView  // Component for displaying the matched component
} from 'vue-router';

export default {
    components: {
        RouterLink, // Register RouterLink for navigation
        RouterView, // Register RouterView for dynamic component rendering
    },
};
</script>

<style scoped>
/* Basic navigation styles */
nav a {
    display: block;
    padding: 10px;
    text-decoration: none;
}

/* Styling for active navigation links */
.v-list-item--active .v-list-item-title {
    font-weight: bold;
}

.v-btn {
  background: transparent;
  --v-activated-opacity: 0;
}

/* Rounded styling for the navigation items */
.v-list-item {
    text-transform: capitalize;
    border-radius: 10px;
}
</style>
