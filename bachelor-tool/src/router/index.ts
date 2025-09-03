import { createRouter, createWebHistory } from 'vue-router'
import HomeView from "@/views/HomeView.vue";
import PartOneView from "@/views/PartOneView.vue";
import PartTwoView from "@/views/PartTwoView.vue";
import DemographicView from "@/views/DemographicView.vue";
import type {Component} from "vue";

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView as Component,
    },
    {
      path: '/one',
      name: 'one',
      component: PartOneView,
    },
    {
      path: '/two',
      name: 'two',
      component: PartTwoView,
    },
    {
      path: '/demographics',
      name: 'demographics',
      component: DemographicView,
    },
  ],
})

export default router
