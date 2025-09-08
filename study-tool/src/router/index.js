import { createRouter, createWebHistory } from "vue-router";

import HomeView from "@/views/HomeView.vue";
import PartOneView from "@/views/PartOneView.vue";
import PartTwoView from "@/views/PartTwoView.vue";
import DemographicView from "@/views/DemographicView.vue";
import FinalView from "@/views/FinalView.vue";

const routes = [
  {
    path: "/",
    name: "home",
    component: HomeView,
  },
  {
    path: "/one",
    name: "one",
    component: PartOneView,
  },
  {
    path: "/two",
    name: "two",
    component: PartTwoView,
  },
  {
    path: "/demographics",
    name: "demographics",
    component: DemographicView,
  },
  {
    path: "/final",
    name: "final",
    component: FinalView,
  },
];

const router = createRouter({
  history: createWebHistory(process.env.BASE_URL),
  routes,
});

export default router;
