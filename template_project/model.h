#include "nnom.h"
#include "weights.h"
nnom_model_t *model;

void initializeModel() {
  model = nnom_model_create();
}
