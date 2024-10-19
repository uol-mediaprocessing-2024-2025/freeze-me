// Import the global CSS and required assets
import './assets/main.css'

import { createApp } from 'vue' // Import the Vue instance
import App from './App.vue' // Import the root component of the app
import router from './router' // Import the router for handling navigation
import '@mdi/font/css/materialdesignicons.min.css'; // Import Material Design Icons

// Import Vuetify components and styles
import 'vuetify/styles'
import { createVuetify } from 'vuetify'
import * as components from 'vuetify/components'
import * as directives from 'vuetify/directives'

// Create an instance of Vuetify for styling and UI components
const vuetify = createVuetify({
    components, // Register Vuetify components
    directives, // Register Vuetify directives
})

// Create the Vue app instance
const app = createApp(App)

// Use the Vuetify plugin for styling and UI functionality
app.use(vuetify)

// Use the Vue Router for navigation
app.use(router)

// Mount the app to the #app element in the HTML
app.mount('#app')
