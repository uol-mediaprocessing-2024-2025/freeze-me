<template>
    <!-- Main application container -->
    <v-app id="app">
        <!-- Sidebar navigation using Vuetify's navigation drawer -->
        <v-navigation-drawer app permanent style="background-color: rgb(30, 50, 92);" class="px-2 text-white">
            <!-- Title and subtitle for the app -->
            <v-list-item title="Simple Web-App" subtitle="Demo"></v-list-item>
            <v-divider></v-divider>

            <!-- Navigation list for routing -->
            <v-list c dense>
                <!-- Link to the main view -->
                <v-list-item class="d-flex" prepend-icon="mdi-home" link to="/">
                    <v-list-item-title>Main</v-list-item-title>
                </v-list-item>

                <!-- Link to the gallery view -->
                <v-list-item class="d-flex" prepend-icon="mdi-image" link to="/gallery">
                    <v-list-item-title>Gallery</v-list-item-title>
                </v-list-item>
            </v-list>
        </v-navigation-drawer>

        <!-- Main content area for rendering views -->
        <v-main style="height: 100vh; overflow-x: hidden;">
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

/* Rounded styling for the navigation items */
.v-list-item {
    text-transform: capitalize;
    border-radius: 10px;
}
</style>
