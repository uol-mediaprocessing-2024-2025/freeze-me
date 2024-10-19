// Import Vue Router components and views
import { createRouter, createWebHistory } from 'vue-router'
import GalleryView from '../views/GalleryView.vue'
import MainView from '../views/MainView.vue'

// Create and configure the router
const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL), // Use HTML5 history mode for clean URLs
  routes: [
    {
      path: '/', // Define the default path for the main view
      name: 'main',
      component: MainView
    },
    {
      path: '/gallery', // Path for gallery view
      name: 'gallery',
      component: GalleryView
    },
  ]
})

// Export the router for use in the Vue app
export default router
