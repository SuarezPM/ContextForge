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

// SchemeBuilder registers the types in this package using the controller-runtime
// helper, which accepts func(*runtime.Scheme) error registration callbacks.
var SchemeBuilder = runtime.NewSchemeBuilder(addKnownTypes)

// AddToScheme adds all types in this package to the provided scheme.
var AddToScheme = SchemeBuilder.AddToScheme

// addKnownTypes registers the API types with their GroupVersion in the scheme.
func addKnownTypes(s *runtime.Scheme) error {
	s.AddKnownTypes(GroupVersion,
		&ApohraContextForgeCluster{},
		&ApohraContextForgeClusterList{},
	)
	metav1.AddToGroupVersion(s, GroupVersion)
	return nil
}
