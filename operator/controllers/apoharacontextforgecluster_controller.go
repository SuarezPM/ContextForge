package controllers

import (
	"context"
	"fmt"

	"k8s.io/apimachinery/pkg/runtime"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	"sigs.k8s.io/controller-runtime/pkg/log"

	contextforgev1alpha1 "github.com/SuarezPM/Apohara_Context_Forge/operator/api/v1alpha1"
)

// ApohraContextForgeClusterReconciler reconciles an ApohraContextForgeCluster object.
type ApohraContextForgeClusterReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=contextforge.apohara.dev,resources=apoharacontextforgeclusters,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=contextforge.apohara.dev,resources=apoharacontextforgeclusters/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=contextforge.apohara.dev,resources=apoharacontextforgeclusters/finalizers,verbs=update
// +kubebuilder:rbac:groups=apps,resources=deployments,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=core,resources=services;configmaps,verbs=get;list;watch;create;update;patch;delete

// Reconcile implements the controller loop for ApohraContextForgeCluster.
// Sprint 1: skeleton only — logs receipt and returns. Real provisioning in Sprint 2.
func (r *ApohraContextForgeClusterReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := log.FromContext(ctx)
	log.Info("reconciled", "cluster", fmt.Sprintf("%s/%s", req.Namespace, req.Name))
	return ctrl.Result{}, nil
}

// SetupWithManager registers the reconciler with the controller-runtime manager.
func (r *ApohraContextForgeClusterReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&contextforgev1alpha1.ApohraContextForgeCluster{}).
		Complete(r)
}
