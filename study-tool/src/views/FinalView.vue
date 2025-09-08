<script setup></script>

<template>
  <main>
    <h1>Vielen Dank, dass Sie an dieser Studie teilgenommen haben.</h1>
    <p>Ihre Antworten wurden erfolgreich gespeichert.</p>
    <p>
      Mit Ihrer Teilnahme leisten Sie einen wichtigen Beitrag zum besseren
      Verständnis der Wahrnehmung von Bildern und Videos.
    </p>
    <p>
      Wenn Sie Fragen oder Anmerkungen zur Studie haben, wenden Sie sich bitte
      an die Studienleitung.
    </p>
  </main>
</template>

<style scoped>
main {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
}

h1 {
  margin: 1em;
}

.dfb_header h2 {
  margin: 0 0 0.25rem 0;
  font-size: 1.125rem;
}

.dfb_header p {
  margin: 0 0 1rem 0;
  color: var(--muted);
  font-size: 0.9375rem;
}

label {
  font-weight: 600;
}

select {
  padding: 0.6rem 0.75rem;
  border-radius: 10px;
  border: var(--border);
  background: #fff;
  font-size: 0.98rem;
}
</style>
