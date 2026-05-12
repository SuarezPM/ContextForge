package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/runtime/schema"
)

// GroupVersion identifies the API group and version for contextforge resources.
var GroupVersion = schema.GroupVersion{
	Group:   "contextforge.apohara.dev",
	Version: "v1alpha1",
}

// SchemeBuilder registers the types in this package.
var SchemeBuilder = &runtime.SchemeBuilder{}

// AddToScheme adds all types in this package to the provided scheme.
var AddToScheme = SchemeBuilder.AddToScheme

func init() {
	SchemeBuilder.Register(
		&ApohraContextForgeCluster{},
		&ApohraContextForgeClusterList{},
	)
	metav1.AddToGroupVersion(scheme, GroupVersion)
}

// scheme is the runtime.Scheme that holds type registrations.
// In production this is wired from main.go via ctrl.NewScheme().
var scheme = runtime.NewScheme()
