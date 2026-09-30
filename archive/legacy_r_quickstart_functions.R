  library('reticulate')
  library('dplyr')
  library(tidyr)
  np <- import('numpy')
  
  load_npy_file <- function(load_path){
    npy_file <- np$load(load_path)
  }
  

  output_weighting <- function(output, data_split) {
    stopifnot(data_split %in% c('train', 'val', 'scoring', 'test'),
              'Provided data_split is not valid. Available options are train, val, scoring, and test.')
    
    num_samples <- nrow(output)
    heating <- matrix(output[, 1:60], nrow = num_samples / self$latlonnum, ncol = self$latlonnum, byrow = TRUE)
    moistening <- matrix(output[, 61:120], nrow = num_samples / self$latlonnum, ncol = self$latlonnum, byrow = TRUE)
    netsw <- matrix(output[, 121], nrow = num_samples / self$latlonnum, ncol = self$latlonnum)
    flwds <- matrix(output[, 122], nrow = num_samples / self$latlonnum, ncol = self$latlonnum)
    precsc <- matrix(output[, 123], nrow = num_samples / self$latlonnum, ncol = self$latlonnum)
    precc <- matrix(output[, 124], nrow = num_samples / self$latlonnum, ncol = self$latlonnum)
    sols <- matrix(output[, 125], nrow = num_samples / self$latlonnum, ncol = self$latlonnum)
    soll <- matrix(output[, 126], nrow = num_samples / self$latlonnum, ncol = self$latlonnum)
    solsd <- matrix(output[, 127], nrow = num_samples / self$latlonnum, ncol = self$latlonnum)
    solld <- matrix(output[, 128], nrow = num_samples / self$latlonnum, ncol = self$latlonnum)
    
    heating <- heating / self$output_scale[['ptend_t']]
    moistening <- moistening / self$output_scale[['ptend_q0001']]
    netsw <- netsw / self$output_scale[['cam_out_NETSW']]
    flwds <- flwds / self$output_scale[['cam_out_FLWDS']]
    precsc <- precsc / self$output_scale[['cam_out_PRECSC']]
    precc <- precc / self$output_scale[['cam_out_PRECC']]
    sols <- sols / self$output_scale[['cam_out_SOLS']]
    soll <- soll / self$output_scale[['cam_out_SOLL']]
    solsd <- solsd / self$output_scale[['cam_out_SOLSD']]
    solld <- solld / self$output_scale[['cam_out_SOLLD']]
    
    if (data_split == 'train') {
      dp <- self$dp_train
    } else if (data_split == 'val') {
      dp <- self$dp_val
    } else if (data_split == 'scoring') {
      dp <- self$dp_scoring
    } else if (data_split == 'test') {
      dp <- self$dp_test
    }
    heating <- heating * dp / self$grav
    moistening <- moistening * dp / self$grav
    
    heating <- heating * self$area_wgt
    moistening <- moistening * self$area_wgt
    netsw <- netsw * self$area_wgt
    flwds <- flwds * self$area_wgt
    precsc <- precsc * self$area_wgt
    precc <- precc * self$area_wgt
    sols <- sols * self$area_wgt
    soll <- soll * self$area_wgt
    solsd <- solsd * self$area_wgt
    solld <- solld * self$area_wgt
    
    heating <- heating * self$target_energy_conv[['ptend_t']]
    moistening <- moistening * self$target_energy_conv[['ptend_q0001']]
    netsw <- netsw * self$target_energy_conv[['cam_out_NETSW']]
    flwds <- flwds * self$target_energy_conv[['cam_out_FLWDS']]
    precsc <- precsc * self$target_energy_conv[['cam_out_PRECSC']]
    precc <- precc * self$target_energy_conv[['cam_out_PRECC']]
    sols <- sols * self$target_energy_conv[['cam_out_SOLS']]
    soll <- soll * self$target_energy_conv[['cam_out_SOLL']]
    solsd <- solsd * self$target_energy_conv[['cam_out_SOLSD']]
    solld <- solld * self$target_energy_conv[['cam_out_SOLLD']]
    
    return(list('ptend_t' = heating,
                'ptend_q0001' = moistening,
                'cam_out_NETSW' = netsw,
                'cam_out_FLWDS' = flwds,
                'cam_out_PRECSC' = precsc,
                'cam_out_PRECC' = precc,
                'cam_out_SOLS' = sols,
                'cam_out_SOLL' = soll,
                'cam_out_SOLSD' = solsd,
                'cam_out_SOLLD' = solld))
  }
  
  
  set_pressure_grid <- function(self,data_split) {
  stopifnot(data_split %in% c('train', 'val', 'scoring', 'test'), 
            'Provided data_split is not valid. Available options are train, val, scoring, and test.')

  if (data_split == 'train') {
    assertthat::assert_that(!is.null(self$input_train))

    state_ps <- self$input_train[, 120] * (self$input_max[['state_ps']] - self$input_min[['state_ps']]) + self$input_mean[['state_ps']]
    state_ps <- matrix(state_ps, nrow = -1, ncol = self$latlonnum)
    pressure_grid_p1 <- as.matrix(self$grid_info[['P0']] * self$grid_info[['hyai']]) %*% matrix(1, nrow = 1, ncol = 1)
    pressure_grid_p2 <- matrix(self$grid_info[['hybi']], nrow = 1, ncol = 1) * state_ps %*% matrix(1, nrow = 1, ncol = ncol(state_ps))
    self$pressure_grid_train <- pressure_grid_p1 + pressure_grid_p2
    self$dp_train <- self$pressure_grid_train[2:61,,] - self$pressure_grid_train[1:60,,]
    self$dp_train <- aperm(self$dp_train, c(3, 1, 2))
    
  } else if (data_split == 'val') {
    assertthat::assert_that(!is.null(self$input_val))

    state_ps <- self$input_val[, 120] * (self$input_max[['state_ps']] - self$input_min[['state_ps']]) + self$input_mean[['state_ps']]
    state_ps <- matrix(state_ps, nrow = -1, ncol = self$latlonnum)
    pressure_grid_p1 <- as.matrix(self$grid_info[['P0']] * self$grid_info[['hyai']]) %*% matrix(1, nrow = 1, ncol = 1)
    pressure_grid_p2 <- matrix(self$grid_info[['hybi']], nrow = 1, ncol = 1) * state_ps %*% matrix(1, nrow = 1, ncol = ncol(state_ps))
    self$pressure_grid_val <- pressure_grid_p1 + pressure_grid_p2
    self$dp_val <- self$pressure_grid_val[2:61,,] - self$pressure_grid_val[1:60,,]
    self$dp_val <- aperm(self$dp_val, c(3, 1, 2))
    
  } else if (data_split == 'scoring') {
    assertthat::assert_that(!is.null(self$input_scoring))

    state_ps <- self$input_scoring[, 120] * (self$input_max[['state_ps']] - self$input_min[['state_ps']]) + self$input_mean[['state_ps']]
    state_ps <- matrix(state_ps, nrow = -1, ncol = self$latlonnum)
    pressure_grid_p1 <- as.matrix(self$grid_info[['P0']] * self$grid_info[['hyai']]) %*% matrix(1, nrow = 1, ncol = 1)
    pressure_grid_p2 <- matrix(self$grid_info[['hybi']], nrow = 1, ncol = 1) * state_ps %*% matrix(1, nrow = 1, ncol = ncol(state_ps))
    self$pressure_grid_scoring <- pressure_grid_p1 + pressure_grid_p2
    self$dp_scoring <- self$pressure_grid_scoring[2:61,,] - self$pressure_grid_scoring[1:60,,]
    self$dp_scoring <- aperm(self$dp_scoring, c(3, 1, 2))
  } else if (data_split == 'test') {
    
    assertthat::assert_that(!is.null(self$input_test))
    state_ps <- self$input_test[, 120] * (self$input_max[['state_ps']] - self$input_min[['state_ps']]) + self$input_mean[['state_ps']]
    state_ps <- matrix(state_ps, nrow = -1, ncol = self$latlonnum)
    pressure_grid_p1 <- as.matrix(self$grid_info[['P0']] * self$grid_info[['hyai']]) %*% matrix(1, nrow = 1, ncol = 1)
    pressure_grid_p2 <- matrix(self$grid_info[['hybi']], nrow = 1, ncol = 1) * state_ps %*% matrix(1, nrow = 1, ncol = ncol(state_ps))
    self$pressure_grid_test <- pressure_grid_p1 + pressure_grid_p2
    self$dp_test <- self$pressure_grid_test[2:61,,] - self$pressure_grid_test[1:60,,]
    self$dp_test <- aperm(self$dp_test, c(3, 1, 2))
  }
}

  
  
  
  reweight_target <- function(self,data_split){
    stopifnot(data_split %in% c('train', 'val', 'scoring', 'test'), 
              'Provided data_split is not valid. Available options are train, val, scoring, and test.')
    if(data_split == 'train'){
      stopif(is.nan(self$target_train))
      self$target_weighted_train <- output_weighting(self$target_train, data_split)
    }else if(data_split == 'val'){
      stopif(is.nan(self$target_val))
      self$target_weighted_val <- output_weighting(self$target_val, data_split)
    }else if(data_split == 'scoring'){
      stopif(is.nan(self$target_scoring))
      self$target_weighted_scoring <- output_weighting(self$target_scoring, data_split)
    }else if(data_split == 'test'){
      stopif(is.nan(self$target_test))
      self$target_weighted_test <- output_weighting(self$target_test, data_split)
    }
  }
  
  
  
  create_metrics_df <- function(data_split) {
    stopifnot(data_split %in% c('train', 'val', 'scoring', 'test'),
              'Provided data_split is not valid. Available options are train, val, scoring, and test.')
    if (data_split == 'train') {
      stopifnot(!is.null(self$preds_weighted_train))
      stopifnot(!is.null(self$target_weighted_train))
      
      for (model_name in self$model_names) {
        df_var <- data.frame('variable' = self$target_vars)
        df_idx <- data.frame('output_idx' = 0:(self$target_feature_len - 1))
        
        for (metric_name in self$metrics_names) {
          current_idx <- 1
          for (target_var in self$target_vars) {
            metric <- self$metrics_dict[[metric_name]](self$preds_weighted_train[[model_name]][[target_var]], self$target_weighted_train[[target_var]])
            df_var[[metric_name]] <- sapply(self$target_vars, function(var) mean(metric))
            df_idx[[metric_name]][current_idx:(current_idx + self$var_lens[[target_var]] - 1)] <- metric
            current_idx <- current_idx + self$var_lens[[target_var]]
          }
        }
        self$metrics_var_train[[model_name]] <- df_var
        self$metrics_idx_train[[model_name]] <- df_idx
      }
    } else if (data_split == 'val') {
    } else if (data_split == 'scoring') {
    } else if (data_split == 'test') {
    }
  }
  
  
  