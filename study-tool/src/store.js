import { reactive } from "vue";

const store = reactive({
  user_path: "",
  apiUrl: "http://localhost:8001",
});

export default store;
